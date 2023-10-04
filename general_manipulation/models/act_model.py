import torch

from torch import nn
from torch.nn import functional as F
from einops import rearrange

from rvt.mvt.attn import (
    Conv2DBlock,
    PreNorm,
    Attention,
    cache_fn,
    DenseBlock,
    FeedForward,
)
from general_manipulation.models.act_cvae import ACTCVAE


class ACTModel(nn.Module):
    def __init__(
        self,
        cfg_dict,
        num_img,
    ):
        """MultiView Transfomer adapted to run ACT."""

        super().__init__()
        self.add_pixel_loc = cfg_dict["add_pixel_loc"]
        self.add_depth = cfg_dict["add_depth"]
        self.add_corr = cfg_dict["add_corr"]
        self.depth = cfg_dict["depth"]
        self.activation = cfg_dict["activation"]
        self.img_feat_dim = cfg_dict["img_feat_dim"]
        self.img_size = cfg_dict["img_size"]
        self.proprio_dim = cfg_dict["proprio_dim"]
        self.im_channels = cfg_dict["im_channels"]
        self.img_patch_size = cfg_dict["img_patch_size"]
        self.attn_dropout = cfg_dict["attn_dropout"]
        self.attn_dim = cfg_dict["attn_dim"]
        self.attn_heads = cfg_dict["attn_heads"]
        self.attn_dim_head = cfg_dict["attn_dim_head"]
        self.num_queries = cfg_dict["num_queries"]
        self.state_dim = cfg_dict["state_dim"]
        self.dim_feedforward = cfg_dict["dim_feedforward"]
        self.num_encoder_layers = cfg_dict["num_encoder_layers"]
        self.num_decoder_layers = cfg_dict["num_decoder_layers"]
        self.normalize_before = cfg_dict["normalize_before"]
        self.kl_weight = cfg_dict["kl_weight"]

        print(f"ACTModel Vars: {vars(self)}")

        self.num_img = num_img

        # patchified input dimensions
        spatial_size = self.img_size // self.img_patch_size  # 220 / 11 = 20

        # 64 img features + 64 proprio features + 64 latent features
        self.input_dim_before_seq = self.im_channels * 3

        # learnable positional encoding
        num_pe_token = spatial_size**2 * self.num_img
        self.pos_embed_encoder = nn.Parameter(
            torch.randn(
                1,
                num_pe_token,
                self.input_dim_before_seq,
            )
        )
        self.pos_embed_decoder = nn.Parameter(
            torch.randn(
                1,
                num_pe_token,
                self.attn_dim,
            )
        )

        inp_img_feat_dim = self.img_feat_dim
        inp_img_feat_dim += 5  # 3 corr + 1 depth + 1 heatmap

        # img input preprocessing encoder
        self.input_preprocess = Conv2DBlock(
            inp_img_feat_dim,
            self.im_channels,
            kernel_sizes=1,
            strides=1,
            norm=None,
            activation=self.activation,
        )
        inp_pre_out_dim = self.im_channels

        # proprio preprocessing encoder
        self.proprio_preprocess = DenseBlock(
            self.proprio_dim,
            self.im_channels,
            norm="group",
            activation=self.activation,
        )

        self.patchify = Conv2DBlock(
            inp_pre_out_dim,
            self.im_channels,
            kernel_sizes=self.img_patch_size,
            strides=self.img_patch_size,
            norm="group",
            activation=self.activation,
            padding=0,
        )

        self.fc_bef_attn = DenseBlock(
            self.input_dim_before_seq,
            self.attn_dim,
            norm=None,
            activation=None,
        )
        self.fc_aft_attn = DenseBlock(
            self.attn_dim,
            self.input_dim_before_seq,
            norm=None,
            activation=None,
        )

        get_attn_attn = lambda: PreNorm(
            self.attn_dim,
            Attention(
                self.attn_dim,
                heads=self.attn_heads,
                dim_head=self.attn_dim_head,
                dropout=self.attn_dropout,
            ),
        )
        get_attn_ff = lambda: PreNorm(self.attn_dim, FeedForward(self.attn_dim))
        get_attn_attn, get_attn_ff = map(cache_fn, (get_attn_attn, get_attn_ff))
        # attention layers
        self.layers = nn.ModuleList([])
        cache_args = {"_cache": False}
        attn_depth = self.depth

        for _ in range(attn_depth):
            self.layers.append(
                nn.ModuleList([get_attn_attn(**cache_args), get_attn_ff(**cache_args)])
            )

        self.query_embed = nn.Embedding(self.num_queries, self.attn_dim)
        self.cvae_encoder = ACTCVAE(
            hidden_dim=self.attn_dim,
            state_dim=self.state_dim,
            num_queries=self.num_queries,
            dropout=self.attn_dropout,
            nhead=self.attn_heads,
            dim_feedforward=self.dim_feedforward,
            num_encoder_layers=self.num_encoder_layers,
            normalize_before=self.normalize_before,
            projected_dim=self.im_channels
        )
        self.decoder = ACTCVAE.build_decoder(
            hidden_dim=self.attn_dim,
            dropout=self.attn_dropout,
            nhead=self.attn_heads,
            dim_feedforward=self.dim_feedforward,
            num_decoder_layers=self.num_decoder_layers,
            normalize_before=self.normalize_before,
        )
        self.action_head = nn.Linear(self.attn_dim, self.state_dim)
        self.is_pad_head = nn.Linear(self.attn_dim, 1)

    def forward(
        self,
        img,
        proprio=None,
        actions=None,
        is_pad=None,
    ):
        """
        :param img: tensor of shape (bs, num_img, img_feat_dim, h, w)
        :param proprio: tensor of shape (bs, proprio_dim)
        :param actions: batch, seq, action_dim
        :param is_pad: batch, seq, 1
        """
        bs, num_img, img_feat_dim, h, w = img.shape
        num_pat_img = h // self.img_patch_size
        assert num_img == self.num_img
        # assert img_feat_dim == self.img_feat_dim
        assert h == w == self.img_size

        # cvae
        if actions is not None:
            latent, (mu, logvar) = self.cvae_encoder(proprio, actions, is_pad)
            training = True
        else:
            latent, (mu, logvar) = self.cvae_encoder(proprio)
            training = False

        img = img.view(bs * num_img, img_feat_dim, h, w)
        # preprocess
        # (bs * num_img, im_channels, h, w)
        d0 = self.input_preprocess(img)

        # (bs * num_img, im_channels, h, w) ->
        # (bs * num_img, im_channels, h / img_patch_strid, w / img_patch_strid) patches
        ins = self.patchify(d0)
        # (bs, im_channels, num_img, h / img_patch_strid, w / img_patch_strid) patches
        ins = (
            ins.view(
                bs,
                num_img,
                self.im_channels,
                num_pat_img,
                num_pat_img,
            )
            .transpose(1, 2)
            .clone()
        )
        # concat proprio
        _, _, _d, _h, _w = ins.shape
        p = self.proprio_preprocess(
            proprio
        )  # [B,7] -> [B,64]
        p = p.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, _d, _h, _w)
        ins = torch.cat([ins, p], dim=1)  # [B, 128, num_img, np, np]

        # concat latent
        latent_processed = (
            latent.unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .repeat(1, 1, _d, _h, _w)
        )
        ins = torch.cat([ins, latent_processed], dim=1)  # [B, 192, num_img, np, np]

        # channel last
        ins = rearrange(ins, "b d ... -> b ... d")  # [B, num_img, np, np, 192]

        # flatten patches into sequence
        ins = rearrange(ins, "b ... d -> b (...) d")  # [B, num_img * np * np, 192]

        # add learable pos encoding
        ins += self.pos_embed_encoder

        x = self.fc_bef_attn(ins)

        # within image self attention
        x = x.reshape(bs * num_img, num_pat_img * num_pat_img, -1)
        for self_attn, self_ff in self.layers[: len(self.layers) // 2]:
            x = self_attn(x) + x
            x = self_ff(x) + x

        x = x.view(bs, num_img * num_pat_img * num_pat_img, -1)
        # attention across images
        for self_attn, self_ff in self.layers[len(self.layers) // 2 :]:
            x = self_attn(x) + x
            x = self_ff(x) + x

        pos = self.pos_embed_decoder.transpose(0, 1).repeat(1, bs, 1)
        memory = x.transpose(0, 1)

        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_embed)
        hs = self.decoder(
            tgt=tgt, memory=memory, pos=pos, query_pos=query_embed
        )
        hs = hs.transpose(1, 2)[0]  # Get last layer output
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)

        if training:
            total_kld, dim_wise_kld, mean_kld = self.cvae_encoder.kl_divergence(
                mu, logvar
            )
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict["l1"] = l1
            loss_dict["kl"] = total_kld[0]
            loss_dict["loss"] = (
                loss_dict["l1"] + loss_dict["kl"] * self.kl_weight
            )
            return loss_dict
        else:
            return a_hat, is_pad_hat, [mu, logvar]

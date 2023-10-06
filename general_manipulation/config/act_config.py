from yacs.config import CfgNode as CN

_C = CN()

_C.proprio_dim = 7  # Proprio as joint abs position.
_C.state_dim = 7
_C.num_queries = 20
_C.dim_feedforward = 4096  # attn_dim (512) * mult (4) * 2 = 4096
_C.num_encoder_layers = 4
_C.num_decoder_layers = 7
_C.normalize_before = False
_C.kl_weight = 10
_C.num_images = 5
_C.debug = True
# Loss weights
_C.act_loss_weight = 100.0  # TODO: Tune
_C.rvt_loss_weight = 1.0  # TODO: Tune


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    return _C.clone()

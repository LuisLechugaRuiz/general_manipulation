# End-to-End Language + Image to Joint Control

## Introduction

In the domain of robotic learning and manipulation, achieving intricate 3D reasoning for tasks such as pick-and-place, object rearrangements, and other sophisticated challenges remains paramount. This project, inspired by previously distinct models like Robotic View Transformer [RVT](https://robotic-view-transformer.github.io/) and Action Chunking [ACT](https://tonyzhaozh.github.io/aloha), takes a step further, converging these methodologies into an integrated architecture.

## Architecture Design

1. **Unified Attention Architecture**:

- **Function**: This architecture integrates image data (augmented with depth information) and textual input for holistic processing.
- **Components**: Initial image parsing is done using a Convolutional Neural Network, which then transitions to a Vision Transformer (ViT) for detailed analysis.

### 2. Dual Decoder System:

- **Shared Encoder Motivation**: We employ a shared encoder for both decoders. The combined loss functions refine the encoder's focus and directionality, leading to better encoded details for predicting joint positions.
- **Keypoint Decoder (Inspired by RVT)**: This processes the ViT encoder output, highlighting critical image regions and providing visual milestones.
- **Joint Position Decoder (Inspired by ACT)**: Informed by the Keypoint Decoder, this extracts insights from the ViT encoder output to anticipate joint positions. It integrates ACT's techniques to mitigate errors common in imitation learning.

## Future Directions

### 3. Real-time Memory Integration:

- **Function**: For improved real-time adaptability and task transition understanding, we're considering embedding memory within our transformer architecture.
- **Design**: Drawing inspiration from the [Vision Transformers Need Registers](https://arxiv.org/pdf/2309.16588.pdf) study, we plan to use discrete memory allocations for each attention head. These would function as registers fed in an autoregressive manner, preserving features for sequential task comprehension. The initial version is developed at [memory branch](https://github.com/LuisLechugaRuiz/RVT/tree/luis/memory)

### 4. Language Model Encoder:

- **Scope**: Our current system operates on unambiguous commands. However, we envision the integration of a Vision-Language Model (VLM) encoder capable of crafting abstract semantic representations. Such an addition would allow the robot to interpret varied command inputs for similar actions, emphasizing the importance of deeper research in reasoning and strategic planning.

## Lessons Learned

For a detailed account of challenges faced, innovations made, and insights gained throughout this project, please refer to our [Lessons Learned](/general_manipulation/docs/leassons_learned.md) documentation.

## Results

The preliminary outcomes of our research and model development can be accessed via the following link: [drive](https://drive.google.com/drive/u/0/folders/143kL1nOz_B534ikPlC3oygQZsLB0B5EG).

Currently, the model demonstrates the capability to manipulate the manipulator based on the encoder's attention cues. Nevertheless, a discrepancy has been observed: the primary area of focus does not consistently align with the designated keypoint. To address this, we are investigating methods to enhance the precision of attention mechanisms. A prospective approach entails integrating language tokens within the self-attention process specific to individual images, as opposed to distributing them across multiple images. This strategy aims to fortify the model's ability to concentrate on salient features from the initial stages.

## Installation

To set up the environment for the project:

1. Begin by following the installation and setup instructions provided in the [Robotic View Transformer (RVT)](https://github.com/NVlabs/RVT#install-rvt) repository.

2. Once completed, navigate to our project directory and execute the following command to install the necessary dependencies:
    `pip3 install .`

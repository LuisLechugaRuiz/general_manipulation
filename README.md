# Robotic Manipulation using Combined ACT and RVT

## Introduction

In the realm of robot learning and manipulation, view-based methods have demonstrated significant success, especially in tasks such as pick-and-place and object rearrangements. Nonetheless, tasks necessitating sophisticated 3D reasoning remain a challenge. This project endeavors to amalgamate two innovative approaches: the Robotic View Transformer [RVT](https://robotic-view-transformer.github.io/) and Action Chunking [ACT](https://tonyzhaozh.github.io/aloha), aiming to tackle these intricate challenges.

## Overview

1. **Robotic View Transformer (RVT)**:
    - **Purpose**: Serving as a global planner, RVT utilizes a multi-view transformer to comprehend the environment and foretell the requisite target pose for executing intricate motion.
    - **Advantage**: As a view-based method, RVT offers scalability and computational efficiency, distinguishing itself from voxel-based alternatives.
2. **Action Chunking (ACT)**:
    - **Purpose**: ACT, acting as the local planner, formulates a trajectory to reach the target pose provided by RVT through the technique of action chunking.
    - **Advantage**: ACT is renowned for its proficiency in predicting high-frequency actions for complex motions.

#### Architecture Design

This project brings together the Robotic View Transformer (RVT) and Action Chunking (ACT), aiming to enhance robotic capabilities for tasks demanding intricate 3D reasoning.

1. **RVT - Perception and Environment Understanding**
    - **Input:** Multi-view images of the scene and natural language instructions.
    - **Process:** RVT employs a multi-view transformer for joint attention across views, understanding the environment, and predicting the end-effector's target pose.
    - **Output:** RVT outputs an 8-dimensional action, which includes the 6-DoF target end-effector pose, 1-DoF gripper state, and a binary collision indicator.

2. **ACT - Action Planning and Execution**
    - **Input:** The input is strategically modified by incorporating multi-view images, enhanced with RVT’s heatmap as an additional channel. This alteration aligns with the rendering technique used by RVT, ensuring consistency and enriched information flow between the global and local planners. Furthermore, a shift is made in proprioceptive information, transitioning from the gripper pose to joint absolute positions, thereby optimizing the input composition for improved trajectory planning.
    - **Process:** The CVAE encoder processes the current joint positions and the target action sequence to infer the style variable ‘z’. This variable, along with the image tokens and the proprioceptive information processed through an MLP, is fed into the joint transformer. The joint transformer features eight self-attention layers; the first four layers are dedicated to processing individual images by restricting attention to tokens within the same image. The subsequent four layers facilitate information propagation and accumulation across different images, synthesizing this diverse information for further action planning.
    - **Output:** A dedicated decoder translates the synthesized information into sequences of actions, forming action chunks aligned with the 7-DoF robot arm's joint absolute positions.

#### Implementation

- **Input Modification and Enriched Rendering:** The implementation strategy involves integrating the heatmap produced by RVT as an additional channel to the virtual images and adjusting the proprioceptive input from the gripper pose to joint absolute positions.

- **Action Chunking Prediction with CVAE and Joint Transformer:** The system leverages the CVAE encoder to process joint positions and target action sequences, inferring the style variable ‘z’. This variable is concatenated with image tokens and processed proprioceptive information, serving as input to the joint transformer. The transformer acts as an encoder combining the virtual images, pointcloud, latent representation from CVAE, and proprioceptive information, culminating in the synthesis of diverse inputs.

#### Future Directions

- **Vision Language Model (VLM) Integration:** Integrating a Vision Language Model (VLM) is a prospective step to further refine the system. VLM will interpret and extract actionable insights from natural language instructions, enabling the robot to comprehend and generalize from linguistic cues. This addition aims to bridge linguistic understanding and visual perception, contributing to the development of a more adaptive and interactive robotic system.

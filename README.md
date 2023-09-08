# Robotic Manipulation using Combined ACT and RVT

## Introduction

In the domain of robot learning and manipulation, view-based methods have achieved remarkable success in tasks like pick-and-place and object rearrangements. However, there's a challenge when it comes to tasks requiring advanced 3D reasoning. This project aims to combine two groundbreaking approaches: Robotic View Transformer (RVT) and Action Chunking (ACT) to address these challenges.

## Overview

1. **Robotic View Transformer (RVT)**:
    - **Purpose**: RVT acts as a global planner. It employs a multi-view transformer to understand the environment and predict the target pose required for a complex motion.
    - **Advantage**: It is view-based, hence scalable and computationally efficient compared to voxel-based methods.
2. **Action Chunking (ACT)**:
    - **Purpose**: ACT is the local planner. Given a target pose from RVT, ACT creates a trajectory to achieve that pose through action chunking.
    - **Advantage**: Efficient high-frequency action predictions for complex motions.

## Architecture Design

### Step 1: RVT - Global Planner

- **Input**: Multi-view images of the scene and a natural language instruction.
- **Process**: The transformer jointly attends over multiple views, aggregates information, and then predicts the robot end-effector pose.
- **Output**: Target pose.

### Step 2: ACT - Local Planner

- **Input**: Target pose from RVT and camera image.
- **Process**: Uses CVAE and action chunking techniques to predict actions on high frequency.
- **Output**: Trajectory or action chunks to achieve the target pose.

## Implementation

1. Integrate RVT to receive multi-view images and produce the desired pose.
2. Pass the output of RVT as an input to ACT.
3. Ensure the end-to-end training of the combined model, allowing joint optimization of RVT and ACT.

## Challenges

- Effective integration of RVT and ACT.
- Ensuring the RVT output contains enough information for ACT predictions.
- Managing computational costs and optimizing hyperparameters.
- Potential risk of overfitting with a complex model.

## Future Steps: Integrating Vision Language Model (VLM)

To further enhance our model, we aim to integrate a Vision Language Model (VLM) as an input to RVT.

**Objective**: The VLM will disambiguate instructions and allow the robot to generalize from natural language instructions. This way, we can leverage both language understanding and image processing.

**How It Works**:
1. **Input**: Natural language instruction along with multi-view images.
2. **VLM Process**: Understands and extracts actionable insights from the instructions.
3. **RVT Process**: Utilizes both the actionable insights from VLM and the images to produce a more refined target pose.
4. **ACT Process**: Continues as before, generating a trajectory to achieve the target pose.

This will pave the way for a more interactive and adaptable robotic system that can understand and execute complex tasks through both visual cues and natural language instructions.
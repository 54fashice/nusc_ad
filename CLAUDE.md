# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains a multimodal autonomous driving system that integrates Large Language Models (LLMs) with 3D perception for end-to-end autonomous driving. The project combines:

1. **DriveLlama** - A custom model that integrates LLaMA with BEVFusion for autonomous driving
2. **LAVIS** - Large Vision-Language models for scene understanding
3. **MMDetection3D** - 3D object detection framework
4. **LLaMA-Factory** - Fine-tuning framework for LLaMA models

## Key Commands

### Data Generation & Preprocessing
```bash
# Generate nuScenes QA dataset
bash data_tools/senna_nusc_converter.sh

# Alternative data generation with specific parameters
python data_tools/senna_nusc_data_converter_api.py nuscenes \
    --root-path /storage/data-acc/nuscenes \
    --out-dir ./data \
    --extra-tag senna_nusc \
    --version v1.0-mini \
    --canbus /storage/data-acc/nuscenes/ \
    --num-workers 32
```

### Training Commands
```bash
# Full fine-tuning with custom configuration
python -m llmtuner.train --config src/configs/full_finetune.json

# LoRA fine-tuning
python -m llmtuner.train --config src/configs/lora_sft.json

# Pre-training with LoRA
python -m llmtuner.train --config src/configs/lora_pretrain.json
```

### MMDetection3D Commands
```bash
# Training
python tools/train.py <CONFIG_FILE>

# Distributed training
bash tools/dist_train.sh <CONFIG_FILE> <GPU_NUM>

# Testing
python tools/test.py <CONFIG_FILE> <CHECKPOINT_FILE>

# Distributed testing
bash tools/dist_test.sh <CONFIG_FILE> <CHECKPOINT_FILE> <GPU_NUM>
```

## Project Structure

### Core Components
- `src/` - Main source code directory
  - `configs/` - Training configurations (JSON files)
  - `llmtuner/` - LLaMA-Factory integration
  - `modules/` - Custom model implementations
- `data_tools/` - Data preprocessing and conversion utilities
- `libs/` - External libraries (LAVIS, MMDetection3D)

### Key Architecture Files
- `src/modules/custom_model.py` - DriveLlama model implementation
- `src/modules/encoders.py` - BEVFusion encoder and vector decorators
- `src/modules/alignment.py` - Q-Former alignment module
- `src/llmtuner/data/nuscenes_dataset.py` - Custom nuScenes dataset loader

## Configuration Files

### Training Configurations
- `full_finetune.json` - Full parameter fine-tuning settings
- `lora_sft.json` - LoRA supervised fine-tuning
- `lora_pretrain.json` - LoRA pre-training configuration
- `sft_config.json` - Standard supervised fine-tuning

### Data Configuration
- `nuscenes_dataloader_config.py` - MMDetection3D data pipeline configuration
- Defines point cloud processing, multi-view image loading, and augmentation strategies

## Model Architecture

### DriveLlama Components
1. **LLaMA Base Model** - Core language model for reasoning
2. **BEVFusion Encoder** - 3D perception backbone with multi-view fusion
3. **Vector Decorator** - Enriches BEV features with vector map data
4. **Q-Former** - Aligns multi-modal features for cross-modal understanding
5. **Trajectory Head** - Outputs driving trajectory predictions

### Key Features
- Multi-modal fusion of LiDAR, camera, and vector map data
- Custom loss functions for trajectory prediction and view alignment
- Freezable components for staged training
- Integration with MMDetection3D data pipeline

## Data Requirements

### nuScenes Dataset
- Point cloud data (LiDAR)
- Multi-view camera images (6 cameras)
- Vector map annotations
- CAN bus data for vehicle state
- Trajectory annotations

### Preprocessing Pipeline
1. Point cloud range filtering and normalization
2. Multi-view image augmentation and normalization
3. 3D bounding box annotation processing
4. Vector map data integration

## Training Strategy

### Three-Stage Training
1. **Pre-training**: Learn basic multimodal alignment
2. **Supervised Fine-tuning**: Task-specific driving behavior learning
3. **Full Fine-tuning**: End-to-end optimization

### Loss Components
- Trajectory prediction loss (MSE)
- View alignment loss (CrossEntropy)
- Language modeling loss (from LLaMA)
- Optional: 3D detection losses

## Development Notes

- Built on MMDetection3D v1.4.0 for 3D perception
- Uses LLaMA-Factory for efficient LLM fine-tuning
- Supports both full parameter and LoRA fine-tuning
- Requires CUDA environment for training
- Point cloud range: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
- Voxel size: [0.2, 0.2, 8]
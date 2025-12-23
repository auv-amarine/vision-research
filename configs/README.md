# Configs

This directory contains YAML configuration files for managing training parameters, model settings, and deployment configurations.

## Purpose

Configuration files provide:
- **Reproducibility**: Exact parameters for each experiment
- **Flexibility**: Easy parameter tuning without code changes
- **Documentation**: Clear record of experiment settings
- **Version Control**: Track configuration changes over time

## Directory Structure

```
configs/
├── data_config.yaml        # Dataset paths and preprocessing settings
├── model_config.yaml       # Model architecture configurations
├── training_config.yaml    # Training hyperparameters
├── inference_config.yaml   # Inference and deployment settings
├── augmentation_config.yaml  # Data augmentation parameters
└── experiments/            # Experiment-specific configs
    ├── baseline.yaml
    ├── ablation_study.yaml
    └── production.yaml
```

## Configuration Files

### 1. Data Configuration (`data_config.yaml`)

Dataset paths, splits, and preprocessing parameters.

```yaml
# data_config.yaml
dataset:
  name: "underwater_objects"
  version: "v1.0"
  
paths:
  train: "data/train"
  val: "data/val"
  test: "data/test"
  annotations: "data/annotations"

classes:
  num_classes: 10
  names:
    - fish
    - coral
    - turtle
    - shark
    - jellyfish
    - starfish
    - crab
    - octopus
    - seaweed
    - debris

preprocessing:
  image_size: [640, 640]
  normalize: true
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]

split:
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  seed: 42
```

### 2. Model Configuration (`model_config.yaml`)

Model architecture and initialization settings.

```yaml
# model_config.yaml
model:
  name: "yolov8"
  variant: "yolov8n"  # n, s, m, l, x
  
architecture:
  backbone: "CSPDarknet"
  neck: "PANet"
  head: "YOLOv8Head"
  
pretrained:
  enabled: true
  weights: "yolov8n.pt"
  freeze_backbone: false
  freeze_layers: 10

input:
  image_size: [640, 640]
  channels: 3

output:
  num_classes: 10
  conf_threshold: 0.25
  iou_threshold: 0.45
  max_detections: 300
```

### 3. Training Configuration (`training_config.yaml`)

Training hyperparameters and optimization settings.

```yaml
# training_config.yaml
training:
  epochs: 100
  batch_size: 16
  num_workers: 4
  
  # Optimization
  optimizer:
    type: "AdamW"
    learning_rate: 0.001
    weight_decay: 0.0005
    momentum: 0.9  # For SGD
    
  scheduler:
    type: "CosineAnnealingLR"
    warmup_epochs: 3
    warmup_lr: 0.0001
    min_lr: 0.00001
    
  # Loss weights
  loss:
    box_loss_weight: 7.5
    cls_loss_weight: 0.5
    obj_loss_weight: 1.0
    
  # Regularization
  regularization:
    dropout: 0.1
    label_smoothing: 0.0
    mixup: 0.0
    cutmix: 0.0

# Hardware
device:
  gpu: true
  gpu_ids: [0]
  mixed_precision: true
  cudnn_benchmark: true

# Checkpointing
checkpoint:
  save_interval: 10
  save_best_only: true
  monitor: "mAP50"
  mode: "max"
  save_dir: "experiments/checkpoints"

# Logging
logging:
  log_interval: 10
  wandb:
    enabled: true
    project: "vision-research"
    entity: "auv-amarine"
  tensorboard:
    enabled: true
    log_dir: "experiments/runs"
    
# Early stopping
early_stopping:
  enabled: true
  patience: 20
  monitor: "mAP50"
  mode: "max"
```

### 4. Augmentation Configuration (`augmentation_config.yaml`)

Data augmentation strategies and parameters.

```yaml
# augmentation_config.yaml
augmentation:
  train:
    # Geometric transformations
    - type: "RandomFlip"
      p: 0.5
      direction: "horizontal"
      
    - type: "RandomRotate"
      p: 0.3
      angle_range: [-15, 15]
      
    - type: "RandomScale"
      p: 0.5
      scale_range: [0.8, 1.2]
      
    - type: "RandomTranslate"
      p: 0.3
      translate_range: [0.1, 0.1]
      
    # Color transformations
    - type: "ColorJitter"
      p: 0.5
      brightness: 0.2
      contrast: 0.2
      saturation: 0.2
      hue: 0.1
      
    - type: "RandomGrayscale"
      p: 0.1
      
    # Underwater-specific augmentations
    - type: "UnderwaterDistortion"
      p: 0.3
      severity: "medium"
      
    - type: "AddWaterParticles"
      p: 0.2
      density: "low"
      
    - type: "SimulateTurbidity"
      p: 0.3
      level_range: [0.1, 0.5]
      
    # Mosaic and mixup
    - type: "Mosaic"
      p: 0.5
      
    - type: "MixUp"
      p: 0.2
      alpha: 0.5
      
  validation:
    # Minimal augmentation for validation
    - type: "Resize"
      size: [640, 640]
      
    - type: "Normalize"
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
      
  test:
    # No augmentation for test
    - type: "Resize"
      size: [640, 640]
      
    - type: "Normalize"
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
```

### 5. Inference Configuration (`inference_config.yaml`)

Deployment and inference settings.

```yaml
# inference_config.yaml
inference:
  model:
    weights: "experiments/checkpoints/best.pt"
    device: "cuda"  # cuda, cpu, or cuda:0
    
  input:
    image_size: [640, 640]
    batch_size: 1
    
  postprocessing:
    conf_threshold: 0.25
    iou_threshold: 0.45
    max_detections: 300
    agnostic_nms: false
    
  output:
    save_predictions: true
    save_dir: "experiments/predictions"
    save_format: "json"  # json, txt, xml
    visualize: true
    show_labels: true
    show_confidence: true
    line_thickness: 2
    
  optimization:
    half_precision: true  # FP16 inference
    tensorrt: false
    onnx: false
    
  video:
    fps: 30
    codec: "mp4v"
    skip_frames: 0
    
  realtime:
    camera_id: 0
    display: true
    record: false
```

## Usage

### Loading Configuration

```python
import yaml
from pathlib import Path

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Load configuration
config = load_config('configs/training_config.yaml')
print(f"Epochs: {config['training']['epochs']}")
print(f"Batch size: {config['training']['batch_size']}")
```

### Merging Configurations

```python
def merge_configs(*config_paths):
    """Merge multiple configuration files."""
    merged = {}
    for path in config_paths:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
            merged.update(config)
    return merged

# Combine multiple configs
config = merge_configs(
    'configs/data_config.yaml',
    'configs/model_config.yaml',
    'configs/training_config.yaml'
)
```

### Using with Command Line

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True,
                   help='Path to configuration file')
parser.add_argument('--epochs', type=int, default=None,
                   help='Override epochs from config')
args = parser.parse_args()

# Load config
config = load_config(args.config)

# Override with command line arguments
if args.epochs:
    config['training']['epochs'] = args.epochs
```

### Using with Dataclass

```python
from dataclasses import dataclass, field
from typing import List
import yaml

@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.001
    device: str = 'cuda'
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict.get('training', {}))

# Usage
config = TrainingConfig.from_yaml('configs/training_config.yaml')
print(f"Training for {config.epochs} epochs")
```

## Best Practices

### 1. Organize by Purpose
- Separate configs for data, model, training, inference
- Create experiment-specific configs in subdirectories
- Use base configs and override for variations

### 2. Use Descriptive Names
```yaml
# Good
learning_rate: 0.001
num_epochs: 100
batch_size: 16

# Avoid
lr: 0.001
n: 100
bs: 16
```

### 3. Document Parameters
```yaml
training:
  epochs: 100  # Number of training epochs
  batch_size: 16  # Batch size per GPU
  learning_rate: 0.001  # Initial learning rate
  
  # Optimizer settings
  optimizer:
    type: "AdamW"  # Options: Adam, AdamW, SGD
    weight_decay: 0.0005  # L2 regularization
```

### 4. Use Environment Variables
```yaml
paths:
  data_root: ${DATA_ROOT:/default/path/to/data}
  output_dir: ${OUTPUT_DIR:experiments/runs}
  
logging:
  wandb:
    api_key: ${WANDB_API_KEY}
```

### 5. Version Control
- Commit all configuration files
- Tag configs used for important experiments
- Document changes in commit messages

### 6. Validation
```python
def validate_config(config: dict) -> bool:
    """Validate configuration parameters."""
    required_keys = ['training', 'model', 'dataset']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key: {key}")
    
    if config['training']['epochs'] <= 0:
        raise ValueError("Epochs must be positive")
        
    if config['training']['batch_size'] <= 0:
        raise ValueError("Batch size must be positive")
        
    return True
```

## Example Workflow

### 1. Create Experiment Config
```yaml
# configs/experiments/underwater_v2.yaml
experiment:
  name: "underwater_v2"
  description: "YOLOv8m with enhanced augmentation"
  
# Import base configs
imports:
  - "../data_config.yaml"
  - "../model_config.yaml"
  - "../training_config.yaml"
  
# Override specific parameters
model:
  variant: "yolov8m"  # Upgrade from yolov8n
  
training:
  epochs: 150  # Train longer
  batch_size: 32  # Larger batch
  
augmentation:
  enhanced: true  # Use more aggressive augmentation
```

### 2. Run Experiment
```bash
python src/training/train.py \
    --config configs/experiments/underwater_v2.yaml \
    --name underwater_v2
```

### 3. Save Final Config
After training, save the exact config used:
```python
import shutil
import yaml

# Save config to experiment directory
exp_dir = f"experiments/runs/{exp_name}"
config_path = f"{exp_dir}/config.yaml"

with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
```

## Tips

- Keep base configurations simple and minimal
- Create specific configs for different scenarios (training, inference, debugging)
- Use YAML anchors and aliases to avoid repetition
- Validate configs before starting long training runs
- Document all non-obvious parameters

---
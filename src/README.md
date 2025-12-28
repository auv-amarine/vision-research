# Source Code

This directory contains production-ready source code for the vision-research project.

## Purpose

The `src/` directory houses well-structured, tested, and optimized code for:
- **Training**: Model training pipelines and optimization
- **Models**: Neural network architectures and model definitions
- **Data**: Dataset loaders, preprocessing, and augmentation
- **Utils**: Helper functions and shared utilities
- **Inference**: Deployment code for real-time detection

## Directory Structure

```
src/
├── training/           # Training modules and pipelines
│   ├── train.py       # Main training script
│   ├── validate.py    # Validation logic
│   └── trainer.py     # Custom trainer class
│
├── models/            # Model architectures
│   ├── yolo.py       # YOLO model definitions
│   ├── backbones.py  # Backbone networks
│   └── heads.py      # Detection heads
│
├── data/              # Data handling
│   ├── dataset.py    # Custom dataset classes
│   ├── augment.py    # Augmentation strategies
│   └── loader.py     # Data loading utilities
│
├── utils/             # Utility functions
│   ├── metrics.py    # Evaluation metrics
│   ├── visualization.py  # Plotting and visualization
│   ├── logger.py     # Logging utilities
│   └── config.py     # Configuration parsing
│
├── dataset/          # For images training
│   ├── data          # Data yaml
|     └── data.yaml   # for training data form yaml document
│   └── roboflow1-by-ezra # Folder training from one person form this group 
|      ├── test
|      ├── train
|      └─ valid
│
└── inference/         # Inference and deployment
    ├── detect.py      # Object detection inference
    ├── tracker.py     # Object tracking
    └── export.py      # Model export (ONNX, TensorRT)
```

## Overview

### 1. Training (`training/`)

Core training functionality for model development.

**Key Components:**
- `train.py`: Main training script with argument parsing and CLI interface
- `validate.py`: Validation and evaluation during training
- `trainer.py`: Optional custom trainer wrapper (for advanced customization)
- `callbacks.py`: Custom callbacks for training monitoring

**Two Training Approaches:**

#### A. Direct YOLO Training (Recommended for Most Cases)
Use Ultralytics YOLO's built-in training - simple and effective:

```python
# train.py
from ultralytics import YOLO
import yaml

def train_yolo(config_path: str):
    """Train YOLO model with custom dataset."""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    model = YOLO(config['model']['weights'])  # yolov8n.pt, yolov8s.pt, etc.
    
    # Train
    results = model.train(
        data=config['data']['yaml_path'],  # dataset.yaml
        epochs=config['training']['epochs'],
        imgsz=config['training']['image_size'],
        batch=config['training']['batch_size'],
        device=config['device']['gpu_ids'],
        project=config['output']['project'],
        name=config['output']['name'],
        patience=config['training'].get('patience', 50),
        save=True,
        plots=True
    )
    
    return results

# Usage
if __name__ == '__main__':
    results = train_yolo('configs/training_config.yaml')
```

#### B. Custom Trainer Wrapper (For Advanced Use Cases)
Only use this when you need custom logic beyond YOLO's capabilities:

```python
# trainer.py
from ultralytics import YOLO
import wandb

class CustomYOLOTrainer:
    """Wrapper around YOLO training for custom workflows."""
    
    def __init__(self, config: dict):
        self.config = config
        self.model = YOLO(config['model']['weights'])
        
    def train(self):
        """Train with custom callbacks and logging."""
        # Initialize W&B
        wandb.init(project="vision-research", config=self.config)
        
        # Train with YOLO
        results = self.model.train(
            **self.config['training_params'],
            callbacks={
                'on_train_epoch_end': self.on_epoch_end,
                'on_val_end': self.on_validation_end
            }
        )
        
        wandb.finish()
        return results
    
    def on_epoch_end(self, trainer):
        """Custom callback after each epoch."""
        # Log custom metrics to W&B
        wandb.log({
            'custom_metric': self.calculate_custom_metric(),
            'epoch': trainer.epoch
        })
```

### 2. Models (`models/`)

Neural network architectures and model definitions.

**Key Components:**
- Model architecture implementations
- Pre-trained model loaders
- Custom layers and modules
- Model factory functions

**Usage Example:**
```python
from src.models.yolo import YOLOv8

model = YOLOv8(num_classes=10, pretrained=True)
predictions = model(images)
```

### 3. Data (`data/`)

Data loading, preprocessing, and augmentation.

**Key Components:**
- Custom dataset classes
- Data augmentation pipelines
- Data loaders with caching
- Preprocessing utilities

**Usage Example:**
```python
from src.data.dataset import UnderwaterDataset
from src.data.augment import get_augmentation_pipeline

dataset = UnderwaterDataset(
    data_dir='data/train',
    augmentation=get_augmentation_pipeline('train')
)
```

### 4. Utils (`utils/`)

Shared utilities and helper functions.

**Key Components:**
- Metrics calculation (mAP, precision, recall)
- Visualization tools
- Configuration management
- Logging and monitoring
- File I/O utilities

**Usage Example:**
```python
from src.utils.metrics import calculate_map
from src.utils.visualization import plot_predictions

map_score = calculate_map(predictions, ground_truth)
plot_predictions(image, predictions, save_path='output.jpg')
```

### 5. Inference (`inference/`)

Deployment and inference code.

**Key Components:**
- Real-time object detection
- Batch inference
- Model export utilities
- Post-processing functions
- Tracking algorithms

**Usage Example:**
```python
from src.inference.detect import ObjectDetector

detector = ObjectDetector(weights='best.pt')
results = detector.predict(image, conf_threshold=0.5)
```

## Code Standards

### 1. Python Style Guide
- Follow PEP 8 style guidelines
- Use type hints for function arguments and returns
- Write docstrings for all functions and classes
- Maximum line length: 100 characters

### 2. Documentation
```python
def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]
    
    Returns:
        IoU score between 0 and 1
    
    Example:
        >>> box1 = np.array([0, 0, 10, 10])
        >>> box2 = np.array([5, 5, 15, 15])
        >>> iou = calculate_iou(box1, box2)
        >>> print(f"IoU: {iou:.3f}")
    """
    # Implementation here
```

### 3. Error Handling
```python
import logging

logger = logging.getLogger(__name__)

def load_model(weights_path: str) -> torch.nn.Module:
    """Load model from weights file."""
    try:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        
        model = torch.load(weights_path)
        logger.info(f"Model loaded successfully from {weights_path}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
```

### 4. Configuration Management
```python
from dataclasses import dataclass
from pathlib import Path

@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.001
    device: str = 'cuda'
    checkpoint_dir: Path = Path('experiments/checkpoints')
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load configuration from YAML file."""
        import yaml
        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
```

## Testing

### Unit Tests
```python
# tests/test_metrics.py
import unittest
from src.utils.metrics import calculate_map

class TestMetrics(unittest.TestCase):
    def test_calculate_map(self):
        predictions = [...]
        ground_truth = [...]
        map_score = calculate_map(predictions, ground_truth)
        self.assertGreater(map_score, 0.0)
        self.assertLessEqual(map_score, 1.0)
```

### Run Tests
```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## Performance Optimization

### 1. GPU Utilization
```python
# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 2. Data Loading
```python
# Use multiple workers for data loading
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2
)
```

### 3. Model Optimization
```python
# Export to ONNX for faster inference
import torch.onnx

torch.onnx.export(
    model,
    dummy_input,
    'model.onnx',
    opset_version=12,
    input_names=['input'],
    output_names=['output']
)
```

## Quick Start: Training YOLO with Custom Dataset

### Step 1: Prepare Your Dataset

Organize your dataset in YOLO format:
```
data/
├── train/
│   ├── images/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── labels/
│       ├── img1.txt
│       └── img2.txt
├── val/
│   ├── images/
│   └── labels/
└── dataset.yaml
```

### Step 2: Create dataset.yaml

```yaml
# data/dataset.yaml
path: /path/to/data  # dataset root dir
train: train/images  # train images (relative to 'path')
val: val/images      # val images (relative to 'path')

# Classes
names:
  0: fish
  1: coral
  2: turtle
  3: shark
  4: jellyfish
```

### Step 3: Create Training Script

```python
# src/training/train.py
import argparse
from ultralytics import YOLO
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='dataset.yaml path')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='model path')
    parser.add_argument('--epochs', type=int, default=100, help='training epochs')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='image size')
    parser.add_argument('--device', type=str, default='0', help='cuda device')
    parser.add_argument('--project', type=str, default='experiments/runs')
    parser.add_argument('--name', type=str, default='train')
    args = parser.parse_args()
    
    # Load model
    model = YOLO(args.model)
    
    # Train model
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=50,
        save=True,
        plots=True,
        val=True
    )
    
    print(f"Training completed! Best model saved to: {results.save_dir}")

if __name__ == '__main__':
    main()
```

### Step 4: Run Training

```bash
# Basic training
python src/training/train.py \
    --data data/dataset.yaml \
    --model yolov8n.pt \
    --epochs 100 \
    --batch 16

# Advanced training with custom settings
python src/training/train.py \
    --data data/dataset.yaml \
    --model yolov8m.pt \
    --epochs 150 \
    --batch 32 \
    --imgsz 640 \
    --device 0,1 \
    --project experiments/runs \
    --name underwater_yolov8m
```

### Step 5: Resume Training (if interrupted)

```python
# Load checkpoint and resume
model = YOLO('experiments/runs/train/weights/last.pt')
model.train(resume=True)
```

## Common Training Scenarios

### 1. Transfer Learning (Fine-tuning)
```python
# Start from pre-trained weights
model = YOLO('yolov8n.pt')  # COCO pre-trained
results = model.train(
    data='data/dataset.yaml',
    epochs=100,
    imgsz=640
)
```

### 2. Training from Scratch
```python
# Start from scratch
model = YOLO('yolov8n.yaml')  # architecture only
results = model.train(
    data='data/dataset.yaml',
    epochs=300,  # needs more epochs
    imgsz=640
)
```

### 3. Multi-GPU Training
```python
# Use multiple GPUs
model = YOLO('yolov8n.pt')
results = model.train(
    data='data/dataset.yaml',
    epochs=100,
    device=[0, 1, 2, 3]  # use 4 GPUs
)
```

### 4. With Data Augmentation
```python
model = YOLO('yolov8n.pt')
results = model.train(
    data='data/dataset.yaml',
    epochs=100,
    hsv_h=0.015,  # HSV-Hue augmentation
    hsv_s=0.7,    # HSV-Saturation
    hsv_v=0.4,    # HSV-Value
    degrees=10,   # rotation
    translate=0.1,
    scale=0.5,
    flipud=0.5,   # vertical flip
    fliplr=0.5,   # horizontal flip
    mosaic=1.0    # mosaic augmentation
)
```

## Migration from Notebooks

When moving code from notebooks to `src/`:

1. **Extract functions**: Convert notebook cells to functions
2. **Add tests**: Write unit tests for new functions
3. **Document**: Add comprehensive docstrings
4. **Refactor**: Improve code quality and structure
5. **Optimize**: Profile and optimize bottlenecks

## YOLO Training Tips

### Best Practices:
- **Start with smaller model** (yolov8n) to verify your dataset works
- **Use pre-trained weights** (transfer learning) for better results
- **Monitor training** with plots in the output directory
- **Validate regularly** to catch overfitting early
- **Save checkpoints** frequently in case of interruptions
- **Use appropriate image size** (640 is standard, but adjust based on your objects)

### Troubleshooting:
- **Low mAP**: Increase epochs, improve data quality, try larger model
- **Overfitting**: Add augmentation, reduce model size, get more data
- **Slow training**: Reduce batch size, use smaller model, check GPU utilization
- **Out of memory**: Reduce batch size or image size

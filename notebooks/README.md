# Notebooks

This directory contains Jupyter notebooks for experimentation, prototyping, and exploratory data analysis.

## Purpose

Notebooks provide an interactive environment for:
- **Data Exploration**: Visualize and analyze datasets
- **Experimentation**: Test new ideas and approaches quickly
- **Visualization**: Create plots and charts for model performance
- **Learning**: Document learning process and insights
- **Prototyping**: Develop and test models before production

## Directory Structure

```
notebooks/
├── data_exploration/       # Dataset analysis and visualization notebooks
├── model_training/         # Training experiments and prototypes
└── evaluation/             # Model evaluation and testing notebooks
```

## Notebook Categories

### 1. Data Exploration
- Dataset statistics and distributions
- Image visualization and augmentation previews
- Class balance analysis
- Annotation format inspection

### 2. Model Training
- Architecture experiments
- Hyperparameter tuning
- Transfer learning trials
- Training pipeline prototypes

### 3. Evaluation
- Model performance analysis
- Error analysis and failure cases
- Comparison between different models
- Inference speed benchmarks

## Naming Convention

Use descriptive names with dates:
```
YYYY-MM-DD_descriptive-name.ipynb

Examples:
2025-01-15_yolov8_dataset_exploration.ipynb
2025-01-20_underwater_augmentation_test.ipynb
2025-02-01_model_comparison_analysis.ipynb
```

## Best Practices

1. **Documentation**
   - Add markdown cells to explain your thought process
   - Include clear section headers
   - Document assumptions and decisions

2. **Code Quality**
   - Keep cells focused and modular
   - Add comments for complex operations
   - Clear all outputs before committing (optional)

3. **Reproducibility**
   - Set random seeds for reproducibility
   - Document dependencies and versions
   - Include data source information

4. **Organization**
   - One notebook per experiment/topic
   - Move production-ready code to `src/`
   - Archive old notebooks in dated folders

## Running Notebooks

### Local Setup
```bash
# Install Jupyter
pip install jupyter notebook

# Launch Jupyter
jupyter notebook

# Or use JupyterLab
pip install jupyterlab
jupyter lab
```

### Using VS Code
- Open `.ipynb` files directly in VS Code
- Select Python kernel
- Run cells interactively

## Converting Notebooks to Scripts

When a notebook is ready for production:
```bash
# Convert to Python script
jupyter nbconvert --to script notebook_name.ipynb

# Move to src/ directory
mv notebook_name.py ../src/training/
```

## Common Imports Template

```python
# Standard libraries
import os
import sys
from pathlib import Path

# Data manipulation
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Computer Vision
import cv2
from PIL import Image

# Deep Learning
import torch
import torch.nn as nn
from torchvision import transforms

# YOLO
from ultralytics import YOLO

# Experiment tracking
import wandb

# Set display options
%matplotlib inline
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
```

## Tips

- Use `%time` or `%%time` magic commands to measure execution time
- Utilize `%debug` for debugging after exceptions
- Use `%load_ext autoreload` for automatic module reloading
- Save checkpoints of long-running experiments

---
# Experiments

This directory stores all experiment results, metrics, model checkpoints, and training artifacts.

## Purpose

The `experiments/` directory maintains a complete record of:
- **Metrics**: Training/validation loss, accuracy, mAP curves
- **Checkpoints**: Saved model weights during training
- **Visualizations**: Plots, graphs, and performance charts
- **Logs**: Training logs and experiment tracking data
- **Results**: Evaluation results and comparison reports

## Directory Structure

```
experiments/
├── runs/                   # Training run outputs
│   ├── exp1/              # Individual experiment folders
│   │   ├── weights/       # Model checkpoints
│   │   ├── logs/          # Training logs
│   │   └── results/       # Evaluation results
│   └── exp2/
│
├── metrics/               # Performance metrics and graphs
│   ├── loss_curves/       # Training and validation loss plots
│   ├── accuracy_plots/    # Accuracy over time
│   └── map_results/       # mAP (mean Average Precision) metrics
│
├── wandb/                 # Weights & Biases tracking data
│   └── run-*/             # Individual W&B run directories
│
└── checkpoints/           # Best model checkpoints
    ├── best.pt            # Best performing model
    ├── last.pt            # Latest checkpoint
    └── checkpoint_epoch_*.pt  # Epoch-specific checkpoints
```

## Experiment Organization

### Naming Convention

Use descriptive experiment names:
```
YYYYMMDD_model_dataset_description

Examples:
20250124_yolov8n_underwater_baseline
20250125_yolov8m_underwater_augmented
20250126_yolov8l_underwater_pretrained
```

### Experiment Folder Contents

Each experiment should contain:
```
exp_name/
├── config.yaml            # Configuration used for this run
├── weights/
│   ├── best.pt           # Best model weights
│   ├── last.pt           # Last epoch weights
│   └── epoch_*.pt        # Intermediate checkpoints
├── logs/
│   ├── train.log         # Training logs
│   └── tensorboard/      # TensorBoard logs
├── results/
│   ├── metrics.json      # Numeric metrics
│   ├── confusion_matrix.png
│   ├── pr_curve.png      # Precision-Recall curve
│   └── predictions/      # Sample predictions
└── README.md             # Experiment notes and findings
```

## Tracking Experiments

### Using Weights & Biases

```python
import wandb

# Initialize W&B
wandb.init(
    project="vision-research",
    name="yolov8_underwater_v1",
    config={
        "learning_rate": 0.001,
        "epochs": 100,
        "batch_size": 16,
        "architecture": "yolov8n"
    }
)

# Log metrics during training
wandb.log({
    "train_loss": loss,
    "val_loss": val_loss,
    "mAP50": map50,
    "epoch": epoch
})

# Log images
wandb.log({"predictions": wandb.Image(img_with_predictions)})

# Finish run
wandb.finish()
```

### Using TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

# Create writer
writer = SummaryWriter(f'experiments/runs/{exp_name}/logs/tensorboard')

# Log scalars
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Loss/val', val_loss, epoch)
writer.add_scalar('Metrics/mAP50', map50, epoch)

# Log images
writer.add_image('Predictions', img_tensor, epoch)

# Log hyperparameters
writer.add_hparams(
    {'lr': lr, 'batch_size': batch_size},
    {'mAP50': map50, 'loss': final_loss}
)

writer.close()
```

## Metrics and Evaluation

### Key Metrics to Track

1. **Training Metrics**
   - Training loss (box, obj, cls)
   - Learning rate
   - Training time per epoch

2. **Validation Metrics**
   - Validation loss
   - Precision
   - Recall
   - mAP@0.5
   - mAP@0.5:0.95

3. **Inference Metrics**
   - FPS (Frames Per Second)
   - Inference time
   - GPU memory usage
   - Model size

### Metrics Visualization

```python
import matplotlib.pyplot as plt
import pandas as pd

# Load metrics
df = pd.read_csv('experiments/runs/exp1/results.csv')

# Plot training curves
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Loss
axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train')
axes[0, 0].plot(df['epoch'], df['val_loss'], label='Validation')
axes[0, 0].set_title('Loss Over Time')
axes[0, 0].legend()

# mAP
axes[0, 1].plot(df['epoch'], df['mAP50'])
axes[0, 1].set_title('mAP@0.5')

# Precision & Recall
axes[1, 0].plot(df['epoch'], df['precision'], label='Precision')
axes[1, 0].plot(df['epoch'], df['recall'], label='Recall')
axes[1, 0].set_title('Precision & Recall')
axes[1, 0].legend()

# F1 Score
axes[1, 1].plot(df['epoch'], df['f1_score'])
axes[1, 1].set_title('F1 Score')

plt.tight_layout()
plt.savefig('experiments/metrics/training_summary.png')
```

## Checkpoint Management

### Saving Checkpoints

```python
def save_checkpoint(model, optimizer, epoch, metrics, path):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")

# Save best model
if val_map > best_map:
    best_map = val_map
    save_checkpoint(
        model, optimizer, epoch, 
        {'mAP50': val_map, 'loss': val_loss},
        'experiments/checkpoints/best.pt'
    )
```

### Loading Checkpoints

```python
def load_checkpoint(model, optimizer, path):
    """Load model checkpoint."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    metrics = checkpoint['metrics']
    print(f"Loaded checkpoint from epoch {epoch}")
    return model, optimizer, epoch, metrics
```

## Experiment Comparison

### Compare Multiple Experiments

```python
import pandas as pd
import seaborn as sns

experiments = [
    'experiments/runs/exp1/results.csv',
    'experiments/runs/exp2/results.csv',
    'experiments/runs/exp3/results.csv'
]

fig, ax = plt.subplots(figsize=(12, 6))

for exp_path in experiments:
    df = pd.read_csv(exp_path)
    exp_name = exp_path.split('/')[2]
    ax.plot(df['epoch'], df['mAP50'], label=exp_name)

ax.set_xlabel('Epoch')
ax.set_ylabel('mAP@0.5')
ax.set_title('Experiment Comparison')
ax.legend()
ax.grid(True)
plt.savefig('experiments/metrics/experiments_comparison.png')
```

## Best Practices

### 1. Document Everything
Create an experiment log:
```markdown
# Experiment: 20250124_yolov8n_underwater_baseline

## Objective
Establish baseline performance for YOLOv8n on underwater dataset

## Configuration
- Model: YOLOv8n
- Dataset: Underwater objects (5000 images)
- Batch size: 16
- Epochs: 100
- Learning rate: 0.001

## Results
- Best mAP@0.5: 0.687
- Training time: 3.5 hours
- Inference speed: 45 FPS

## Observations
- Model converges well after epoch 60
- Some confusion between fish species
- Good performance on large objects

## Next Steps
- Try YOLOv8m for better accuracy
- Add more data augmentation
- Fine-tune on hard examples
```

### 2. Clean Up Old Experiments
```bash
# Archive experiments older than 30 days
find experiments/runs/ -type d -mtime +30 -exec tar -czf {}.tar.gz {} \;
```

### 3. Backup Important Results
```bash
# Backup best models
rsync -av experiments/checkpoints/ backup/checkpoints/

# Backup to cloud storage
aws s3 sync experiments/checkpoints/ s3://bucket/vision-research/checkpoints/
```

### 4. Version Control for Experiments
- Tag important experiments in git
- Store experiment configs in version control
- Don't commit large checkpoint files (use .gitignore)

## Generating Reports

### Automatic Report Generation

```python
def generate_experiment_report(exp_name: str):
    """Generate HTML report for an experiment."""
    from jinja2 import Template
    
    # Load metrics
    df = pd.read_csv(f'experiments/runs/{exp_name}/results.csv')
    
    # Generate plots
    generate_plots(df, exp_name)
    
    # Create report
    template = Template(report_template)
    html = template.render(
        exp_name=exp_name,
        best_map=df['mAP50'].max(),
        final_loss=df['val_loss'].iloc[-1],
        plots=['loss.png', 'map.png', 'pr_curve.png']
    )
    
    with open(f'experiments/runs/{exp_name}/report.html', 'w') as f:
        f.write(html)
```

## Troubleshooting

- **Disk space full**: Archive old experiments, delete intermediate checkpoints
- **W&B not syncing**: Check internet connection, run `wandb sync` manually
- **Metrics not logging**: Verify logger is properly initialized
- **Checkpoint corrupt**: Keep multiple backup checkpoints

---
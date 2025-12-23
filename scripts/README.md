# Scripts

This directory contains shell scripts for automating various tasks in the vision-research pipeline.

## Purpose

Shell scripts automate repetitive tasks including:
- **Training**: Execute training jobs with specific configurations
- **Data Processing**: Batch data preprocessing and augmentation
- **Model Conversion**: Convert models for deployment (ONNX, TensorRT)
- **Inference**: Run batch inference on datasets
- **Maintenance**: Clean up temporary files, organize outputs

## Common Scripts

### Training Scripts
```bash
# Train with specific config
./scripts/train.sh --config configs/yolov8_config.yaml

# Multi-GPU training
./scripts/train_multigpu.sh --gpus 0,1,2,3

# Resume from checkpoint
./scripts/resume_training.sh --checkpoint experiments/checkpoints/last.pt
```

### Inference Scripts
```bash
# Batch inference on video
./scripts/inference.sh --source data/videos/ --weights best.pt

# Real-time inference
./scripts/realtime_inference.sh --camera 0
```

### Data Processing Scripts
```bash
# Preprocess dataset
./scripts/data_preprocessing.sh --input data/raw/ --output data/processed/

# Split dataset
./scripts/split_dataset.sh --ratio 0.8:0.1:0.1
```

## Script Template

Here's a basic template for creating new scripts:

```bash
#!/bin/bash

# Script: script_name.sh
# Description: Brief description of what this script does
# Usage: ./script_name.sh [options]

set -e  # Exit on error
set -u  # Exit on undefined variable

# Default values
CONFIG_FILE="configs/default.yaml"
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--config FILE] [--verbose]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Main logic
echo "Starting script with config: $CONFIG_FILE"

# Your commands here
python src/training/train.py --config "$CONFIG_FILE"

echo "Script completed successfully!"
```

## Best Practices

### 1. Script Headers
- Add shebang line: `#!/bin/bash`
- Include description and usage
- Document required arguments

### 2. Error Handling
```bash
set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure
```

### 3. Logging
```bash
LOG_FILE="logs/$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE")
exec 2>&1
```

### 4. Argument Parsing
- Provide default values
- Validate required arguments
- Include help option

### 5. Make Scripts Executable
```bash
chmod +x scripts/*.sh
```

## Common Patterns

### Run Python Script with Error Handling
```bash
python src/training/train.py --config "$CONFIG" || {
    echo "Training failed!"
    exit 1
}
```

### Loop Through Files
```bash
for file in data/images/*.jpg; do
    echo "Processing: $file"
    python src/inference/detect.py --source "$file"
done
```

### Conditional Execution
```bash
if [ -f "experiments/checkpoints/best.pt" ]; then
    echo "Using existing checkpoint"
    CHECKPOINT="experiments/checkpoints/best.pt"
else
    echo "Starting from scratch"
    CHECKPOINT=""
fi
```

### Parallel Processing
```bash
# Run multiple jobs in parallel
for i in {0..3}; do
    python train.py --gpu $i --fold $i &
done
wait  # Wait for all background jobs
```

## Environment Variables

Scripts can read from `.env` file:
```bash
# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi
```

## Scheduling Scripts

### Using Cron
```bash
# Edit crontab
crontab -e

# Run training daily at 2 AM
0 2 * * * cd /path/to/vision-research && ./scripts/train.sh
```

### Using tmux/screen
```bash
# Start background session
tmux new -s training
./scripts/train.sh
# Detach: Ctrl+B, then D

# Reattach later
tmux attach -t training
```

## Example Scripts

### 1. `train.sh` - Training Script
```bash
#!/bin/bash
set -e

CONFIG=${1:-configs/training_config.yaml}
GPU=${2:-0}

echo "Starting training with config: $CONFIG on GPU: $GPU"

CUDA_VISIBLE_DEVICES=$GPU python src/training/train.py \
    --config "$CONFIG" \
    --project experiments/runs \
    --name "$(date +%Y%m%d_%H%M%S)"

echo "Training completed!"
```

### 2. `inference.sh` - Batch Inference
```bash
#!/bin/bash
set -e

WEIGHTS=${1:-experiments/checkpoints/best.pt}
SOURCE=${2:-data/test/}
OUTPUT=${3:-experiments/predictions/}

python src/inference/detect.py \
    --weights "$WEIGHTS" \
    --source "$SOURCE" \
    --output "$OUTPUT" \
    --conf-thres 0.25 \
    --iou-thres 0.45

echo "Inference results saved to: $OUTPUT"
```

### 3. `setup_env.sh` - Environment Setup
```bash
#!/bin/bash
set -e

echo "Setting up vision-research environment..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Setup directories
mkdir -p data/{train,val,test}
mkdir -p experiments/{runs,checkpoints,metrics}

echo "Environment setup complete!"
```

## Troubleshooting

- **Permission denied**: Run `chmod +x script.sh`
- **Command not found**: Check if Python/dependencies are in PATH
- **Script hangs**: Check for missing input or infinite loops
- **GPU not detected**: Verify CUDA_VISIBLE_DEVICES environment variable

---
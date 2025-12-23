#!/bin/bash

#training Script for YOLO Object Detection
#this script reads configuration from configs/training_config.yaml
#executes training using src/training/train.py

set -e  

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' 

#def
CONFIG_FILE="configs/training_config.yaml"
OVERRIDE_ARGS=""

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --config FILE       Configuration file (default: configs/training_config.yaml)"
    echo "  --data FILE         Override dataset.yaml path"
    echo "  --model FILE        Override model weights"
    echo "  --epochs NUM        Override number of epochs"
    echo "  --batch NUM         Override batch size"
    echo "  --device ID         Override GPU device (0, 0,1,2,3, or cpu)"
    echo "  --name NAME         Override experiment name"
    echo "  --resume            Resume from last checkpoint"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Use default config"
    echo "  $0 --epochs 150 --batch 32            # Override specific parameters"
    echo "  $0 --config configs/custom.yaml       # Use custom config"
    echo "  $0 --resume                           # Resume training"
    exit 1
}

#parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --data)
            OVERRIDE_ARGS="$OVERRIDE_ARGS --data $2"
            shift 2
            ;;
        --model)
            OVERRIDE_ARGS="$OVERRIDE_ARGS --model $2"
            shift 2
            ;;
        --epochs)
            OVERRIDE_ARGS="$OVERRIDE_ARGS --epochs $2"
            shift 2
            ;;
        --batch)
            OVERRIDE_ARGS="$OVERRIDE_ARGS --batch $2"
            shift 2
            ;;
        --device)
            OVERRIDE_ARGS="$OVERRIDE_ARGS --device $2"
            shift 2
            ;;
        --name)
            OVERRIDE_ARGS="$OVERRIDE_ARGS --name $2"
            shift 2
            ;;
        --resume)
            OVERRIDE_ARGS="$OVERRIDE_ARGS --resume"
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
done

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Configuration file not found: $CONFIG_FILE${NC}"
    exit 1
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  YOLO Training Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Configuration:${NC} $CONFIG_FILE"
if [ ! -z "$OVERRIDE_ARGS" ]; then
    echo -e "${GREEN}Overrides:${NC}$OVERRIDE_ARGS"
fi
echo -e "${BLUE}========================================${NC}"
echo ""

mkdir -p experiments/runs
mkdir -p experiments/checkpoints

if [ ! -f "src/training/train.py" ]; then
    echo -e "${RED}Error: Training script not found: src/training/train.py${NC}"
    exit 1
fi

echo -e "${GREEN}Starting training...${NC}"
echo ""

python src/training/train.py \
    --config "$CONFIG_FILE" \
    $OVERRIDE_ARGS

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Training completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "Results saved to: experiments/runs/"
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}  Training failed!${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi

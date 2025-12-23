#!/bin/bash

#inference Script for YOLO Object Detection
#this script reads configuration from configs/inference_config.yaml
#executes inference using src/inference/detect.py (soon)

set -e 

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' 

CONFIG_FILE="configs/inference_config.yaml"
OVERRIDE_ARGS=""

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --config FILE       Configuration file (default: configs/inference_config.yaml)"
    echo "  --weights FILE      Override model weights path"
    echo "  --source PATH       Override source (directory, video, image, or 0 for webcam)"
    echo "  --conf FLOAT        Override confidence threshold (0.0-1.0)"
    echo "  --iou FLOAT         Override IoU threshold (0.0-1.0)"
    echo "  --device ID         Override device (0, 0,1,2,3, or cpu)"
    echo "  --save-dir PATH     Override save directory"
    echo "  --no-save           Don't save results"
    echo "  --show              Display results in real-time"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                            # Use default config"
    echo "  $0 --source data/test/images                  # Inference on directory"
    echo "  $0 --source video.mp4 --conf 0.5              # Inference on video"
    echo "  $0 --source 0 --show                          # Real-time webcam inference"
    echo "  $0 --weights experiments/runs/train/weights/best.pt --source test.jpg"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --weights)
            OVERRIDE_ARGS="$OVERRIDE_ARGS --weights $2"
            shift 2
            ;;
        --source)
            OVERRIDE_ARGS="$OVERRIDE_ARGS --source $2"
            shift 2
            ;;
        --conf)
            OVERRIDE_ARGS="$OVERRIDE_ARGS --conf $2"
            shift 2
            ;;
        --iou)
            OVERRIDE_ARGS="$OVERRIDE_ARGS --iou $2"
            shift 2
            ;;
        --device)
            OVERRIDE_ARGS="$OVERRIDE_ARGS --device $2"
            shift 2
            ;;
        --save-dir)
            OVERRIDE_ARGS="$OVERRIDE_ARGS --save-dir $2"
            shift 2
            ;;
        --no-save)
            OVERRIDE_ARGS="$OVERRIDE_ARGS --no-save"
            shift
            ;;
        --show)
            OVERRIDE_ARGS="$OVERRIDE_ARGS --show"
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
echo -e "${BLUE}  YOLO Inference Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Configuration:${NC} $CONFIG_FILE"
if [ ! -z "$OVERRIDE_ARGS" ]; then
    echo -e "${GREEN}Overrides:${NC}$OVERRIDE_ARGS"
fi
echo -e "${BLUE}========================================${NC}"
echo ""

mkdir -p experiments/predictions

if [ ! -f "src/inference/detect.py" ]; then
    echo -e "${YELLOW}Warning: Inference script not found: src/inference/detect.py${NC}"
    echo -e "${YELLOW}Using YOLO CLI instead...${NC}"
    echo ""
    
    #fallback to yOLO CLI
    #extract values from config (simplified approach)
    WEIGHTS=$(grep "weights:" "$CONFIG_FILE" | awk '{print $2}' | tr -d '"')
    SOURCE=$(grep "source:" "$CONFIG_FILE" | awk '{print $2}' | tr -d '"')
    CONF=$(grep "conf_threshold:" "$CONFIG_FILE" | awk '{print $2}')
    
    echo -e "${GREEN}Running inference with YOLO CLI...${NC}"
    yolo detect predict \
        model="$WEIGHTS" \
        source="$SOURCE" \
        conf="$CONF" \
        save=True \
        $OVERRIDE_ARGS
    
    exit 0
fi

echo -e "${GREEN}Starting inference...${NC}"
echo ""

#detect.py (soon)
python src/inference/detect.py \
    --config "$CONFIG_FILE" \
    $OVERRIDE_ARGS

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Inference completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "Results saved to: experiments/predictions/"
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}  Inference failed!${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi

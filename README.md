<div align="center">
<img width="1200" height="275" alt="PPBanner" src="public/assets/PASTI PINTAR BANNER.jpg" />
</div>

# Vision Research

> Research Repository of AUV Vision Amarine

Vision Research Repository is a comprehensive research platform dedicated to advancing computer vision capabilities for Autonomous Underwater Vehicles (AUV). This repository focuses on object detection, image classification, and other computer vision techniques to enable autonomous navigation and operation of AUV systems.

## Library, Framework & Technology Used

Vision Research utilizes cutting-edge AI frameworks and libraries specifically optimized for image classification and object detection tasks. To support AUV autonomous operations, we integrate NVIDIA frameworks compatible with Jetson hardware platforms deployed on our AUV systems.

### AI Library & Framework
- **YOLO** (You Only Look Once) - Real-time object detection system
- **PyTorch** - Deep learning framework for model development and training
- **TensorFlow** - Machine learning platform for production deployment
- **Ultralytics** - YOLOv.x implementation and training pipeline
- **Weights & Biases (wandb)** - Experiment tracking and model performance monitoring
- **NVIDIA TensorRT** - High-performance deep learning inference optimization
- **CUDA** - Parallel computing platform for GPU acceleration

## Repo Structure

```
vision-research/
├── paper/                      # Research papers and documentation
    *will implement as soon as possible*
│   ├── object_detection/       # Papers on object detection methodologies
│   ├── underwater_vision/      # Underwater computer vision research
│   └── autonomous_systems/     # AUV autonomy-related papers
│
├── notebooks/                  # Jupyter notebooks for experimentation
    *will implement as soon as possible*
│   ├── data_exploration/       # Dataset analysis and visualization
│   ├── model_training/         # Training experiments and prototypes
│   └── evaluation/             # Model evaluation and testing
│
├── scripts/                    # Shell scripts for automation
│   ├── train.sh                # Training execution scripts
│   ├── inference.sh            # Inference pipeline scripts
│   └── data_preprocessing.sh   # Data preparation scripts
│
├── src/                        # Source code for production
    *will implement as soon as possible*
│   ├── training/               # Training modules and pipelines
│   │   ├── train.py
│   │   └── validate.py
│   ├── models/                 # Model architectures
│   ├── data/                   # Data loaders and augmentation
│   ├── utils/                  # Utility functions and helpers
│   └── inference/              # Inference and deployment code
│
├── experiments/                # Experiment results and metrics
│   ├── runs/                   # Training run outputs
│   ├── metrics/                # Loss, accuracy, mAP graphs
│   ├── wandb/                  # Weights & Biases logs
│   └── checkpoints/            # Model checkpoints
│
├── configs/                    # Configuration files
│   ├── data_config.yaml        # Dataset configuration
│   ├── model_config.yaml       # Model hyperparameters
│   └── training_config.yaml    # Training parameters
│
├── requirements.txt            # Python dependencies (soon)
├── .env.template               # Environment variables template
├── LICENSE                     # Apache License
└── README.md                   # This file
```

## Getting Started

In this repository we need to install a few dependencies for easier setup in our workspace.

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- NVIDIA Jetson (for deployment on AUV hardware)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/auv-amarine/vision-research.git
   cd vision-research
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Copy .env.template to your workspace
   ```bash
   cp .env.template .env
   ```
   Edit `.env` file with your configuration (API keys, paths, etc.)

4. **Verify installation:**
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}')"
   python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
   ```

## Quick Overview of Directories

### `paper/`
Contains research papers, academic publications, and documentation related to underwater computer vision, object detection algorithms, and AUV autonomous navigation systems. This directory serves as a knowledge base for the theoretical foundations of our work.

### `notebooks/`
Jupyter notebooks for rapid prototyping and experimentation. Use these notebooks to:
- Explore and visualize datasets
- Test different model architectures
- Conduct preliminary training experiments
- Evaluate model performance before production deployment

### `scripts/`
Shell scripts for automating repetitive tasks such as:
- Training multiple models with different configurations
- Batch inference on video datasets
- Data preprocessing and augmentation pipelines
- Model conversion and optimization for deployment

### `src/`
Production-ready source code organized into modules:
- **training/**: Training loops, validation procedures, and optimization routines
- **models/**: Custom model architectures and pretrained model loaders
- **data/**: Dataset loaders, augmentation strategies, and preprocessing
- **utils/**: Helper functions, logging utilities, and common operations
- **inference/**: Deployment code for real-time object detection on AUV

### `experiments/`
Stores all experiment results including:
- Training metrics (loss curves, accuracy plots)
- Evaluation metrics (mAP, precision, recall)
- Model checkpoints and weights
- Weights & Biases integration for experiment tracking and comparison

### `configs/`
YAML configuration files for reproducible experiments:
- Dataset paths and augmentation parameters
- Model architecture specifications
- Training hyperparameters (learning rate, batch size, epochs)
- Inference settings and deployment configurations

## Usage Examples

### Training a Model
```bash
# Using Python script
python src/training/train.py --config configs/training_config.yaml

# Using shell script
bash scripts/train.sh
```

### Running Inference
```bash
python src/inference/detect.py --weights experiments/checkpoints/best.pt --source data/test_video.mp4
```

### Tracking Experiments
```bash
# Initialize wandb
wandb login

# Experiments are automatically logged to wandb during training
# View results at https://wandb.ai/your-username/vision-research
```

## License

This project is under Apache License 2.0. See [LICENSE](LICENSE) for more details.

## Support & Contact

Support, contact, or get in touch with our maintainers of this repository:
- **Wildan Aziz**: [Website](https://wildanaziz.vercel.app/) | [GitHub](https://github.com/wildanaziz)
- **Ezra**: [GitHub](https://github.com/ezra1702)

## Acknowledgments

- **AUV Amarine Team** - For continuous support and collaboration on autonomous underwater vehicle development
- **Ultralytics** - For the excellent YOLOv8 implementation and documentation
- **PyTorch Community** - For the robust deep learning framework
- **Weights & Biases** - For providing powerful experiment tracking tools
- **NVIDIA** - For Jetson platform support and optimization tools

---

<div align="center">

**Built with attention for AUV Amarine**

*blubub blubub* 

</div>
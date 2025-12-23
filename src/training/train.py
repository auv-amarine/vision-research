#!/usr/bin/env python3
"""
YOLO Training Script with Configuration Support

This script trains YOLO models using configurations from configs/training_config.yaml
Can be executed directly or via scripts/train.sh

Usage:
    python src/training/train.py --config configs/training_config.yaml
    python src/training/train.py --config configs/training_config.yaml --epochs 150 --batch 32
    bash scripts/train.sh
    bash scripts/train.sh --epochs 150 --name my_experiment
"""

import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_args_with_config(config: dict, args: argparse.Namespace) -> dict:
    """Merge command-line arguments with configuration file."""
    
    # Override model settings
    if args.model:
        config['model']['weights'] = args.model
    
    # Override dataset settings
    if args.data:
        config['data']['yaml_path'] = args.data
    
    # Override training parameters
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch:
        config['training']['batch_size'] = args.batch
    if args.imgsz:
        config['training']['image_size'] = args.imgsz
    
    # Override device settings
    if args.device:
        config['device']['gpu_ids'] = args.device
    
    # Override output settings
    if args.project:
        config['output']['project'] = args.project
    if args.name:
        config['output']['name'] = args.name
    if args.resume:
        config['advanced']['resume'] = True
    
    return config


def print_training_info(config: dict):
    """Print training configuration information."""
    print("\n" + "="*60)
    print("  YOLO Training Configuration")
    print("="*60)
    print(f"Model:        {config['model']['weights']}")
    print(f"Dataset:      {config['data']['yaml_path']}")
    print(f"Epochs:       {config['training']['epochs']}")
    print(f"Batch Size:   {config['training']['batch_size']}")
    print(f"Image Size:   {config['training']['image_size']}")
    print(f"Device:       {config['device']['gpu_ids']}")
    print(f"Output:       {config['output']['project']}/{config['output']['name']}")
    print("="*60 + "\n")


def train_yolo(config: dict):
    """
    Train YOLO model with the provided configuration.
    
    Args:
        config: Configuration dictionary loaded from YAML
    
    Returns:
        Training results object
    """
    
    print_training_info(config)
    
    # Initialize Weights & Biases if enabled
    wandb_enabled = config.get('logging', {}).get('wandb', {}).get('enabled', False)
    if wandb_enabled:
        if not WANDB_AVAILABLE:
            print("WARNING: wandb is enabled in config but not installed. Disabling wandb logging.")
            wandb_enabled = False
        else:
            wandb_config = config['logging']['wandb']
            
            wandb_api_key = os.getenv('WANDB_API_KEY')
            wandb_entity = os.getenv('WANDB_ENTITY') or wandb_config.get('entity')
            wandb_project = os.getenv('WANDB_PROJECT') or wandb_config['project']
            wandb_name = wandb_config.get('name') or config['output']['name']
            
            # Set API key if provided in environment
            if wandb_api_key:
                os.environ['WANDB_API_KEY'] = wandb_api_key
            
            print(f"Initializing Weights & Biases...")
            print(f"  Project: {wandb_project}")
            print(f"  Entity: {wandb_entity or 'default'}")
            print(f"  Name: {wandb_name}\n")
            
            try:
                wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=wandb_name,
                    config={
                        'model': config['model'],
                        'training': config['training'],
                        'data': config['data'],
                        'device': config['device']
                    }
                )
            except Exception as e:
                print(f"WARNING: Failed to initialize wandb: {e}")
                print("Continuing training without wandb logging.\n")
                wandb_enabled = False
    
    if config['device']['gpu_ids'] != 'cpu':
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available, using CPU instead")
            config['device']['gpu_ids'] = 'cpu'
        else:
            print(f"Using GPU(s): {config['device']['gpu_ids']}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"PyTorch Version: {torch.__version__}\n")
    
    # Load model
    print(f"Loading model: {config['model']['weights']}")
    model = YOLO(config['model']['weights'])
    
    # Prepare training arguments
    train_args = {
        # Data
        'data': config['data']['yaml_path'],
        
        # Training
        'epochs': config['training']['epochs'],
        'batch': config['training']['batch_size'],
        'imgsz': config['training']['image_size'],
        'patience': config['training']['patience'],
        
        # Optimizer
        'optimizer': config['training']['optimizer'],
        'lr0': config['training']['lr0'],
        'lrf': config['training']['lrf'],
        'momentum': config['training']['momentum'],
        'weight_decay': config['training']['weight_decay'],
        
        # Augmentation
        'hsv_h': config['training']['hsv_h'],
        'hsv_s': config['training']['hsv_s'],
        'hsv_v': config['training']['hsv_v'],
        'degrees': config['training']['degrees'],
        'translate': config['training']['translate'],
        'scale': config['training']['scale'],
        'shear': config['training']['shear'],
        'perspective': config['training']['perspective'],
        'flipud': config['training']['flipud'],
        'fliplr': config['training']['fliplr'],
        'mosaic': config['training']['mosaic'],
        'mixup': config['training']['mixup'],
        'copy_paste': config['training']['copy_paste'],
        
        # Device
        'device': config['device']['gpu_ids'],
        'workers': config['device']['workers'],
        
        # Output
        'project': config['output']['project'],
        'name': config['output']['name'],
        'exist_ok': config['output']['exist_ok'],
        'save': config['output']['save'],
        'save_period': config['output']['save_period'],
        'plots': config['output']['plots'],
        'verbose': config['output']['verbose'],
        
        # Validation
        'val': config['validation']['val'],
        'split': config['validation']['split'],
        'save_json': config['validation']['save_json'],
        'save_hybrid': config['validation']['save_hybrid'],
        
        # Advanced
        'cache': config['advanced']['cache'],
        'rect': config['advanced']['rect'],
        'resume': config['advanced']['resume'],
        'amp': config['advanced']['amp'],
        'fraction': config['advanced']['fraction'],
        'profile': config['advanced']['profile'],
        'freeze': config['advanced']['freeze'],
        'multi_scale': config['advanced']['multi_scale'],
        'overlap_mask': config['advanced']['overlap_mask'],
        'mask_ratio': config['advanced']['mask_ratio'],
        'dropout': config['advanced']['dropout'],
        'val_period': config['advanced']['val_period'],
    }
    
    print("Starting training...\n")
    results = model.train(**train_args)
    
    # Log final results to wandb
    if wandb_enabled:
        try:
            wandb.log({
                "final/mAP50": results.results_dict.get('metrics/mAP50(B)', 0),
                "final/mAP50-95": results.results_dict.get('metrics/mAP50-95(B)', 0),
                "final/precision": results.results_dict.get('metrics/precision(B)', 0),
                "final/recall": results.results_dict.get('metrics/recall(B)', 0),
            })
            
            if results.save_dir:
                best_model = os.path.join(results.save_dir, 'weights', 'best.pt')
                if os.path.exists(best_model):
                    wandb.save(best_model)
                    print("Model uploaded to Weights & Biases")
            
            wandb.finish()
            print("Weights & Biases logging completed\n")
        except Exception as e:
            print(f"Warning: Failed to log to wandb: {e}")
            wandb.finish()
    
    print("\n" + "="*60)
    print("  Training Completed!")
    print("="*60)
    print(f"Best model saved to: {results.save_dir}")
    print(f"Results saved to: {config['output']['project']}/{config['output']['name']}")
    if wandb_enabled:
        wandb_entity = os.getenv('WANDB_ENTITY') or config.get('logging', {}).get('wandb', {}).get('entity', 'your-entity')
        wandb_project = os.getenv('WANDB_PROJECT') or config.get('logging', {}).get('wandb', {}).get('project', 'vision-research')
        print(f"Wandb logs: https://wandb.ai/{wandb_entity}/{wandb_project}")
    print("="*60 + "\n")
    
    return results


def main():
    """Main training function."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Train YOLO model with custom dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Train with default config
        python src/training/train.py --config configs/training_config.yaml

        # Override specific parameters
        python src/training/train.py --config configs/training_config.yaml --epochs 150 --batch 32

        # Use custom model and dataset
        python src/training/train.py --config configs/training_config.yaml --model yolov8m.pt --data custom.yaml

        # Resume training
        python src/training/train.py --config configs/training_config.yaml --resume
        """
    )
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to training configuration YAML file')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to dataset.yaml (overrides config)')
    parser.add_argument('--model', type=str, default=None,
                       help='Model weights path (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--batch', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--imgsz', type=int, default=None,
                       help='Image size (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                       help='CUDA device (overrides config)')
    parser.add_argument('--project', type=str, default=None,
                       help='Project directory (overrides config)')
    parser.add_argument('--name', type=str, default=None,
                       help='Experiment name (overrides config)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from last checkpoint')
    
    args = parser.parse_args()
    
    if not Path(args.config).exists():
        print(f"Error: Configuration file not found: {args.config}")
        return
    
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    config = merge_args_with_config(config, args)
    
    train_yolo(config)


if __name__ == '__main__':
    main()

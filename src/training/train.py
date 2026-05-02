#!/usr/bin/env python3
"""
YOLO Training Script with Configuration Support, W&B Callbacks, Artifacts & Sweeps

This script trains YOLO models using configurations from configs/training_config.yaml.
It supports Weights & Biases integration via the official Ultralytics callback
(add_wandb_callback), W&B Artifacts for model versioning, and Hyperparameter Sweeps.

Usage:
    # Standard training
    python src/training/train.py --config configs/training_config.yaml
    python src/training/train.py --config configs/training_config.yaml --epochs 150 --batch 32

    # Hyperparameter sweep
    python src/training/train.py --config configs/training_config.yaml --sweep --sweep_count 20

    # Via shell script
    bash scripts/train.sh
    bash scripts/train.sh --epochs 150 --name my_experiment
"""

import argparse
import yaml
import os
from pathlib import Path
from typing import Dict, Any

from ultralytics import YOLO
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Weights & Biases Setup ---
WANDB_AVAILABLE = False
add_wandb_callback = None
wandb = None

try:
    import wandb as _wandb
    from wandb.integration.ultralytics import add_wandb_callback as _add_wandb_callback

    wandb = _wandb
    add_wandb_callback = _add_wandb_callback
    WANDB_AVAILABLE = True
except Exception as e:
    print(f"Warning: wandb or ultralytics integration not available: {e}")
    print("Install with: pip install wandb")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def merge_args_with_config(config: dict, args: argparse.Namespace) -> dict:
    """Merge command-line arguments with configuration file."""
    # Override model settings
    if args.model:
        config["model"]["weights"] = args.model

    # Override dataset settings
    if args.data:
        config["data"]["yaml_path"] = args.data

    # Override training parameters
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch:
        config["training"]["batch_size"] = args.batch
    if args.imgsz:
        config["training"]["image_size"] = args.imgsz

    # Override device settings
    if args.device:
        config["device"]["gpu_ids"] = args.device

    # Override output settings
    if args.project:
        config["output"]["project"] = args.project
    if args.name:
        config["output"]["name"] = args.name
    if args.resume:
        config["advanced"]["resume"] = True

    return config


def print_training_info(config: dict):
    """Print training configuration information."""
    print("\n" + "=" * 60)
    print("  YOLO Training Configuration")
    print("=" * 60)
    print(f"Model:        {config['model']['weights']}")
    print(f"Dataset:      {config['data']['yaml_path']}")
    print(f"Epochs:       {config['training']['epochs']}")
    print(f"Batch Size:   {config['training']['batch_size']}")
    print(f"Image Size:   {config['training']['image_size']}")
    print(f"Device:       {config['device']['gpu_ids']}")
    print(f"Output:       {config['output']['project']}/{config['output']['name']}")
    print("=" * 60 + "\n")


def build_train_args(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build Ultralytics training arguments from the custom config dict."""
    return {
        # Data
        "data": config["data"]["yaml_path"],
        # Training
        "epochs": config["training"]["epochs"],
        "batch": config["training"]["batch_size"],
        "imgsz": config["training"]["image_size"],
        "patience": config["training"]["patience"],
        # Optimizer
        "optimizer": config["training"]["optimizer"],
        "lr0": config["training"]["lr0"],
        "lrf": config["training"]["lrf"],
        "momentum": config["training"]["momentum"],
        "weight_decay": config["training"]["weight_decay"],
        "label_smoothing": config["training"]["label_smoothing"],
        # Augmentation
        "hsv_h": config["training"]["hsv_h"],
        "hsv_s": config["training"]["hsv_s"],
        "hsv_v": config["training"]["hsv_v"],
        "degrees": config["training"]["degrees"],
        "translate": config["training"]["translate"],
        "scale": config["training"]["scale"],
        "shear": config["training"]["shear"],
        "perspective": config["training"]["perspective"],
        "flipud": config["training"]["flipud"],
        "fliplr": config["training"]["fliplr"],
        "mosaic": config["training"]["mosaic"],
        "mixup": config["training"]["mixup"],
        "copy_paste": config["training"]["copy_paste"],
        # Device
        "device": config["device"]["gpu_ids"],
        "workers": config["device"]["workers"],
        # Output
        "project": config["output"]["project"],
        "name": config["output"]["name"],
        "exist_ok": config["output"]["exist_ok"],
        "save": config["output"]["save"],
        "save_period": config["output"]["save_period"],
        "plots": config["output"]["plots"],
        "verbose": config["output"]["verbose"],
        # Validation
        "val": config["validation"]["val"],
        "split": config["validation"]["split"],
        "save_json": config["validation"]["save_json"],
        # Advanced
        "cache": config["advanced"]["cache"],
        "rect": config["advanced"]["rect"],
        "resume": config["advanced"]["resume"],
        "amp": config["advanced"]["amp"],
        "fraction": config["advanced"]["fraction"],
        "profile": config["advanced"]["profile"],
        "freeze": config["advanced"]["freeze"],
        "multi_scale": config["advanced"]["multi_scale"],
        "overlap_mask": config["advanced"]["overlap_mask"],
        "mask_ratio": config["advanced"]["mask_ratio"],
        "dropout": config["advanced"]["dropout"],
    }


def apply_wandb_config_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply hyperparameter overrides from wandb.config.
    This is used ONLY when the script is executed inside a W&B Sweep.
    """
    if not (WANDB_AVAILABLE and wandb.run and wandb.run.sweep_id is not None):
        return config

    sweep = wandb.config
    overrides = {
        "epochs": ("training", "epochs"),
        "batch_size": ("training", "batch_size"),
        "image_size": ("training", "image_size"),
        "lr0": ("training", "lr0"),
        "lrf": ("training", "lrf"),
        "momentum": ("training", "momentum"),
        "weight_decay": ("training", "weight_decay"),
        "optimizer": ("training", "optimizer"),
        "patience": ("training", "patience"),
        "dropout": ("advanced", "dropout"),
        "mixup": ("training", "mixup"),
        "mosaic": ("training", "mosaic"),
        "copy_paste": ("training", "copy_paste"),
    }

    for key, (section, cfg_key) in overrides.items():
        if key in sweep:
            old_val = config[section][cfg_key]
            new_val = sweep[key]
            # Safety: skip if W&B returns a dict back for a scalar key
            if isinstance(new_val, dict):
                print(f"  [Sweep Warning] skipping '{key}': received dict instead of scalar")
                continue
            config[section][cfg_key] = new_val
            print(f"  [Sweep Override] {section}.{cfg_key}: {old_val} -> {new_val}")

    return config


def log_model_artifact(model_path: Path, alias: str = "model"):
    """Log a model checkpoint as a W&B Artifact."""
    if not (WANDB_AVAILABLE and wandb.run):
        return

    if not model_path.exists():
        print(f"  Warning: Model path not found: {model_path}")
        return

    artifact_name = (
        f"{wandb.run.name}-{alias}"
        if wandb.run.name
        else f"run-{wandb.run.id}-{alias}"
    )
    artifact = wandb.Artifact(name=artifact_name, type="model")
    artifact.add_file(str(model_path))
    wandb.log_artifact(artifact)
    print(f"  Artifact logged: {artifact_name} ({model_path.name})")


def train_yolo(config: Dict[str, Any]):
    """
    Train YOLO model with the provided configuration.

    Args:
        config: Configuration dictionary loaded from YAML.

    Returns:
        Training results object.
    """
    print_training_info(config)

    # --- Device Check ---
    if config["device"]["gpu_ids"] != "cpu":
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available, using CPU instead")
            config["device"]["gpu_ids"] = "cpu"
        else:
            print(f"Using GPU(s): {config['device']['gpu_ids']}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"PyTorch Version: {torch.__version__}\n")

    # --- Initialize Weights & Biases ---
    wandb_enabled = config.get("logging", {}).get("wandb", {}).get("enabled", False)
    if wandb_enabled and WANDB_AVAILABLE:
        wandb_cfg = config["logging"]["wandb"]
        wandb_entity = os.getenv("WANDB_ENTITY") or wandb_cfg.get("entity")
        wandb_project = os.getenv("WANDB_PROJECT") or wandb_cfg["project"]
        wandb_name = wandb_cfg.get("name") or config["output"]["name"]

        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key:
            os.environ["WANDB_API_KEY"] = wandb_api_key

        print("Initializing Weights & Biases...")
        print(f"  Project: {wandb_project}")
        print(f"  Entity:  {wandb_entity or 'default'}")
        print(f"  Name:    {wandb_name}\n")

        try:
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_name,
                job_type="training",
                config=config,
            )
        except Exception as e:
            print(f"WARNING: Failed to initialize wandb: {e}")
            print("Continuing training without wandb logging.\n")
            wandb_enabled = False

    # --- Apply Sweep Overrides (if active) ---
    if wandb_enabled:
        config = apply_wandb_config_overrides(config)

    # --- Load Model ---
    print(f"Loading model: {config['model']['weights']}")
    model = YOLO(config["model"]["weights"])

    # --- Attach official W&B callback for full metrics & media tracking ---
    if wandb_enabled and WANDB_AVAILABLE and add_wandb_callback is not None:
        add_wandb_callback(model, enable_model_checkpoints=False)
        print("Official W&B Ultralytics callback attached (full metrics + media tracking)\n")

    # --- Build Training Arguments ---
    train_args = build_train_args(config)

    # --- Training ---
    print("Starting training...\n")
    results = model.train(**train_args)

    # --- Post-Training Validation (optional) ---
    if config.get("validation", {}).get("val", True):
        print("\nRunning post-training validation...")
        val_device = config["device"]["gpu_ids"]
        if val_device != "cpu" and not torch.cuda.is_available():
            val_device = "cpu"
        try:
            model.val(data=config["data"]["yaml_path"], device=val_device)
        except Exception as e:
            print(f"Warning: Post-training validation failed: {e}")

    # --- Log Model Artifacts ---
    if wandb_enabled and results.save_dir:
        save_dir = Path(results.save_dir)
        best_pt = save_dir / "weights" / "best.pt"
        last_pt = save_dir / "weights" / "last.pt"

        print("\nLogging model artifacts to W&B...")
        log_model_artifact(best_pt, alias="best")
        log_model_artifact(last_pt, alias="last")

    # --- Finish W&B Run ---
    if wandb_enabled and WANDB_AVAILABLE:
        wandb.finish()
        print("Weights & Biases logging completed\n")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  Training Completed!")
    print("=" * 60)
    print(f"Best model saved to: {results.save_dir}")
    print(f"Results saved to: {config['output']['project']}/{config['output']['name']}")
    if wandb_enabled:
        wandb_entity = (
            os.getenv("WANDB_ENTITY")
            or config.get("logging", {}).get("wandb", {}).get("entity", "your-entity")
        )
        wandb_project = (
            os.getenv("WANDB_PROJECT")
            or config.get("logging", {}).get("wandb", {}).get("project", "vision-research")
        )
        print(f"Wandb logs: https://wandb.ai/{wandb_entity}/{wandb_project}")
    print("=" * 60 + "\n")

    return results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train YOLO model with custom dataset, W&B callbacks, artifacts & sweeps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard training
  python src/training/train.py --config configs/training_config.yaml

  # Override parameters
  python src/training/train.py --config configs/training_config.yaml --epochs 150 --batch 32

  # Resume training
  python src/training/train.py --config configs/training_config.yaml --resume

  # Hyperparameter sweep
  python src/training/train.py --config configs/training_config.yaml --sweep --sweep_count 20
        """,
    )

    parser.add_argument(
        "--config", type=str, required=True, help="Path to training configuration YAML file"
    )
    parser.add_argument(
        "--data", type=str, default=None, help="Path to dataset.yaml (overrides config)"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Model weights path (overrides config)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Number of training epochs (overrides config)"
    )
    parser.add_argument(
        "--batch", type=int, default=None, help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=None, help="Image size (overrides config)"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="CUDA device (overrides config)"
    )
    parser.add_argument(
        "--project", type=str, default=None, help="Project directory (overrides config)"
    )
    parser.add_argument(
        "--name", type=str, default=None, help="Experiment name (overrides config)"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from last checkpoint"
    )
    parser.add_argument(
        "--sweep", action="store_true", help="Run a W&B hyperparameter sweep"
    )
    parser.add_argument(
        "--sweep_count", type=int, default=10, help="Number of sweep runs to execute"
    )

    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"Error: Configuration file not found: {args.config}")
        return

    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    config = merge_args_with_config(config, args)

    # --- Sweep Mode ---
    if args.sweep:
        if not WANDB_AVAILABLE:
            print("Error: wandb is required for sweeps. Install with: pip install wandb")
            return

        wandb_cfg = config.get("logging", {}).get("wandb", {})
        wandb_entity = os.getenv("WANDB_ENTITY") or wandb_cfg.get("entity")
        wandb_project = os.getenv("WANDB_PROJECT") or wandb_cfg.get("project", "vision-research")

        sweep_config = config.get("sweep")
        if not sweep_config:
            print("Warning: No 'sweep' section found in config. Using default sweep config.")
            sweep_config = {
                "method": "random",
                "metric": {"name": "metrics/mAP50-95(B)", "goal": "maximize"},
                "parameters": {
                    "lr0": {"distribution": "uniform", "min": 0.001, "max": 0.1},
                    "batch_size": {"values": [8, 16, 32]},
                    "image_size": {"values": [416, 512, 640]},
                },
            }

        print("Initializing W&B Sweep...")
        sweep_id = wandb.sweep(
            sweep=sweep_config, project=wandb_project, entity=wandb_entity
        )
        print(f"Sweep ID: {sweep_id}\n")

        def sweep_train():
            """Function executed by each W&B sweep agent."""
            cfg = load_config(args.config)
            cfg = merge_args_with_config(cfg, args)
            # Force-enable W&B for sweep runs
            cfg.setdefault("logging", {})
            cfg["logging"].setdefault("wandb", {})
            cfg["logging"]["wandb"]["enabled"] = True
            cfg["logging"]["wandb"]["project"] = wandb_project
            if wandb_entity:
                cfg["logging"]["wandb"]["entity"] = wandb_entity
            train_yolo(cfg)

        wandb.agent(sweep_id, function=sweep_train, count=args.sweep_count)
    else:
        train_yolo(config)


if __name__ == "__main__":
    main()

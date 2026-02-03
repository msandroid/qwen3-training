#!/usr/bin/env python3
"""
MS-SWIFT Training Script for Qwen3 Fine-tuning

This script provides a Python wrapper for MS-SWIFT training,
supporting both LoRA and full fine-tuning methods.

Requirements:
- NVIDIA GPU with CUDA support
- ms-swift >= 3.0.0
- transformers >= 4.40.0

Usage:
    python train_msswift.py --config config.yaml
    python train_msswift.py --model Qwen/Qwen3-4B-Instruct --dataset train.jsonl
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    yaml = None


def check_cuda():
    """Check if CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
            return True
        else:
            print("Warning: CUDA not available. Training will be very slow.")
            return False
    except ImportError:
        print("Error: PyTorch not installed.")
        return False


def check_msswift():
    """Check if MS-SWIFT is installed."""
    try:
        import swift
        print(f"MS-SWIFT available")
        return True
    except ImportError:
        print("Error: MS-SWIFT not installed.")
        print("Install with: pip install ms-swift -U")
        return False


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    if yaml is None:
        raise ImportError("PyYAML not installed. Install with: pip install pyyaml")
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_swift_command(
    model: str,
    dataset: str,
    output_dir: str,
    val_dataset: Optional[str] = None,
    tuner_type: str = "lora",
    lora_rank: int = 16,
    lora_alpha: int = 32,
    learning_rate: float = 1e-4,
    num_train_epochs: int = 3,
    batch_size: int = 2,
    max_length: int = 512,
    gradient_accumulation_steps: int = 4,
    save_steps: int = 500,
    logging_steps: int = 10,
    warmup_ratio: float = 0.1,
    use_flash_attn: bool = True,
    quant_bits: Optional[int] = None,
    deepspeed: Optional[str] = None,
    **kwargs
) -> list[str]:
    """Build MS-SWIFT training command."""
    
    cmd = [
        "swift", "sft",
        "--model", model,
        "--dataset", dataset,
        "--output_dir", output_dir,
        "--tuner_type", tuner_type,
        "--torch_dtype", "bfloat16",
        "--learning_rate", str(learning_rate),
        "--num_train_epochs", str(num_train_epochs),
        "--per_device_train_batch_size", str(batch_size),
        "--max_length", str(max_length),
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        "--save_steps", str(save_steps),
        "--logging_steps", str(logging_steps),
        "--warmup_ratio", str(warmup_ratio),
        "--save_total_limit", "3",
        "--dataloader_num_workers", "4",
    ]
    
    # Validation dataset
    if val_dataset:
        cmd.extend(["--val_dataset", val_dataset])
    
    # LoRA parameters
    if tuner_type == "lora":
        cmd.extend([
            "--lora_rank", str(lora_rank),
            "--lora_alpha", str(lora_alpha),
            "--target_modules", "all-linear",
            "--lora_dropout", "0.05",
        ])
    
    # Flash Attention
    if use_flash_attn:
        cmd.extend(["--attn_impl", "flash_attn"])
    
    # Quantization (QLoRA)
    if quant_bits:
        cmd.extend(["--quant_bits", str(quant_bits)])
    
    # DeepSpeed
    if deepspeed:
        cmd.extend(["--deepspeed", deepspeed])
    
    return cmd


def run_training(cmd: list[str], dry_run: bool = False) -> bool:
    """Execute training command."""
    
    cmd_str = " \\\n  ".join(cmd)
    print(f"\n{'='*60}")
    print("Training Command:")
    print(f"{'='*60}")
    print(cmd_str)
    print(f"{'='*60}\n")
    
    if dry_run:
        print("Dry run mode - command not executed.")
        return True
    
    # Set environment variables
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    
    # Run training
    process = subprocess.run(cmd, env=env)
    return process.returncode == 0


def create_config_template(output_path: str):
    """Create a sample configuration file."""
    
    config = {
        "model": {
            "name": "Qwen/Qwen3-4B-Instruct",
            "torch_dtype": "bfloat16",
        },
        "data": {
            "train_dataset": "data/train.jsonl",
            "val_dataset": "data/valid.jsonl",
            "max_length": 512,
        },
        "lora": {
            "tuner_type": "lora",
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": "all-linear",
        },
        "training": {
            "num_train_epochs": 3,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 1e-4,
            "warmup_ratio": 0.1,
            "save_steps": 500,
            "logging_steps": 10,
        },
        "output": {
            "output_dir": "outputs/my-model",
            "save_total_limit": 3,
        },
        "advanced": {
            "use_flash_attn": True,
            "quant_bits": None,  # Set to 4 for QLoRA
            "deepspeed": None,  # Set to "zero2" or "zero3" for distributed
        },
    }
    
    if yaml is None:
        # Write as JSON if PyYAML not available
        output_path = output_path.replace(".yaml", ".json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"Config template saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="MS-SWIFT Training Script for Qwen3 Fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with config file
  python train_msswift.py --config config.yaml
  
  # Train with command line arguments
  python train_msswift.py \\
    --model Qwen/Qwen3-4B-Instruct \\
    --dataset train.jsonl \\
    --output-dir outputs/my-model \\
    --lora-rank 16 \\
    --epochs 3
  
  # Generate config template
  python train_msswift.py --create-config config.yaml
  
  # Dry run (show command without executing)
  python train_msswift.py --config config.yaml --dry-run
        """
    )
    
    # Config file
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--create-config", type=str, help="Create config template at path")
    
    # Model
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct",
                        help="Model name or path")
    
    # Data
    parser.add_argument("--dataset", type=str, help="Path to training dataset (JSONL)")
    parser.add_argument("--val-dataset", type=str, help="Path to validation dataset")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    
    # LoRA
    parser.add_argument("--tuner-type", type=str, default="lora",
                        choices=["lora", "full"], help="Fine-tuning method")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    
    # Training
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gradient-accumulation", type=int, default=4,
                        help="Gradient accumulation steps")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="outputs/model",
                        help="Output directory")
    parser.add_argument("--save-steps", type=int, default=500, help="Save checkpoint every N steps")
    
    # Advanced
    parser.add_argument("--quant-bits", type=int, choices=[4, 8],
                        help="Quantization bits for QLoRA")
    parser.add_argument("--deepspeed", type=str, choices=["zero2", "zero3"],
                        help="DeepSpeed config")
    parser.add_argument("--no-flash-attn", action="store_true",
                        help="Disable Flash Attention")
    
    # Execution
    parser.add_argument("--dry-run", action="store_true",
                        help="Show command without executing")
    parser.add_argument("--skip-checks", action="store_true",
                        help="Skip CUDA and MS-SWIFT checks")
    
    args = parser.parse_args()
    
    # Create config template
    if args.create_config:
        create_config_template(args.create_config)
        return
    
    # Load config file if provided
    config = {}
    if args.config:
        config = load_config(args.config)
    
    # Merge config with command line arguments
    model = args.model
    dataset = args.dataset
    val_dataset = args.val_dataset
    output_dir = args.output_dir
    
    if config:
        model = config.get("model", {}).get("name", model)
        dataset = config.get("data", {}).get("train_dataset", dataset)
        val_dataset = config.get("data", {}).get("val_dataset", val_dataset)
        output_dir = config.get("output", {}).get("output_dir", output_dir)
    
    # Validate required arguments
    if not dataset:
        parser.error("--dataset is required (or specify in config file)")
    
    # Check prerequisites
    if not args.skip_checks:
        print("Checking prerequisites...")
        if not check_cuda():
            print("Warning: Continuing without CUDA (training will be slow)")
        if not check_msswift():
            sys.exit(1)
    
    # Build training parameters
    train_params = {
        "model": model,
        "dataset": dataset,
        "output_dir": output_dir,
        "val_dataset": val_dataset,
        "tuner_type": args.tuner_type,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "save_steps": args.save_steps,
        "use_flash_attn": not args.no_flash_attn,
        "quant_bits": args.quant_bits,
        "deepspeed": args.deepspeed,
    }
    
    # Override with config values
    if config:
        lora_config = config.get("lora", {})
        training_config = config.get("training", {})
        advanced_config = config.get("advanced", {})
        
        train_params.update({
            "tuner_type": lora_config.get("tuner_type", train_params["tuner_type"]),
            "lora_rank": lora_config.get("rank", train_params["lora_rank"]),
            "lora_alpha": lora_config.get("alpha", train_params["lora_alpha"]),
            "num_train_epochs": training_config.get("num_train_epochs", train_params["num_train_epochs"]),
            "batch_size": training_config.get("per_device_train_batch_size", train_params["batch_size"]),
            "learning_rate": training_config.get("learning_rate", train_params["learning_rate"]),
            "gradient_accumulation_steps": training_config.get("gradient_accumulation_steps", train_params["gradient_accumulation_steps"]),
            "save_steps": training_config.get("save_steps", train_params["save_steps"]),
            "max_length": config.get("data", {}).get("max_length", train_params["max_length"]),
            "use_flash_attn": advanced_config.get("use_flash_attn", train_params["use_flash_attn"]),
            "quant_bits": advanced_config.get("quant_bits", train_params["quant_bits"]),
            "deepspeed": advanced_config.get("deepspeed", train_params["deepspeed"]),
        })
    
    # Build and run command
    cmd = build_swift_command(**train_params)
    
    print(f"\n{'='*60}")
    print("MS-SWIFT Training")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Dataset: {dataset}")
    print(f"Output: {output_dir}")
    print(f"Method: {train_params['tuner_type']}")
    if train_params['tuner_type'] == 'lora':
        print(f"LoRA Rank: {train_params['lora_rank']}")
    print(f"Epochs: {train_params['num_train_epochs']}")
    print(f"{'='*60}")
    
    success = run_training(cmd, dry_run=args.dry_run)
    
    if success and not args.dry_run:
        print(f"\n{'='*60}")
        print("Training completed successfully!")
        print(f"Model saved to: {output_dir}")
        print(f"{'='*60}")
        print("\nNext steps:")
        print("1. Merge adapter: swift export --model ... --adapter ... --merge_lora true")
        print("2. Convert to GGUF: python convert_hf_to_gguf.py ...")
        print("3. Evaluate: python evaluate_gguf.py --model ...")
    elif not args.dry_run:
        print("\nTraining failed. Check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

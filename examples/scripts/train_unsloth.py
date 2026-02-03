#!/usr/bin/env python3
"""
Unsloth Training Script for Qwen3 Fine-tuning

Unsloth provides 2x faster training with 70% less VRAM usage.
This script supports LoRA fine-tuning with automatic 4-bit quantization.

Requirements:
- NVIDIA GPU with CUDA support
- unsloth >= 2024.8
- transformers >= 4.40.0

Usage:
    python train_unsloth.py --dataset train.jsonl --output-dir outputs/model
    python train_unsloth.py --config config.json
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def check_dependencies():
    """Check if required packages are installed."""
    missing = []
    
    try:
        import torch
        if not torch.cuda.is_available():
            print("Warning: CUDA not available. Unsloth requires CUDA.")
    except ImportError:
        missing.append("torch")
    
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        missing.append("unsloth")
    
    try:
        from datasets import load_dataset
    except ImportError:
        missing.append("datasets")
    
    try:
        from trl import SFTTrainer
    except ImportError:
        missing.append("trl")
    
    if missing:
        print(f"Missing dependencies: {missing}")
        print("Install with:")
        print('  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"')
        print("  pip install --no-deps xformers trl peft accelerate bitsandbytes")
        return False
    
    return True


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_messages_to_text(example: dict, tokenizer) -> dict:
    """Convert messages format to text for training."""
    messages = example.get("messages", [])
    
    # Use tokenizer's chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    else:
        # Fallback: manual formatting
        text_parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                text_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                text_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                text_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        text = "\n".join(text_parts)
    
    return {"text": text}


def train_model(
    model_name: str = "Qwen/Qwen3-4B-Instruct",
    dataset_path: str = "train.jsonl",
    output_dir: str = "outputs/model",
    max_seq_length: int = 512,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    num_train_epochs: int = 3,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 1e-4,
    warmup_steps: int = 100,
    save_steps: int = 500,
    logging_steps: int = 10,
    load_in_4bit: bool = True,
    use_gradient_checkpointing: bool = True,
):
    """Train model using Unsloth."""
    
    import torch
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments
    
    print(f"\n{'='*60}")
    print("Unsloth Training")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")
    print(f"LoRA Rank: {lora_rank}")
    print(f"Epochs: {num_train_epochs}")
    print(f"{'='*60}\n")
    
    # Load model with 4-bit quantization
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=load_in_4bit,
    )
    
    # Apply LoRA
    print("Applying LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth" if use_gradient_checkpointing else False,
        random_state=42,
    )
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    # Format dataset
    print("Formatting dataset...")
    dataset = dataset.map(
        lambda x: format_messages_to_text(x, tokenizer),
        remove_columns=dataset.column_names,
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        save_total_limit=3,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",  # Disable wandb/tensorboard
    )
    
    # Create trainer
    print("Starting training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=training_args,
    )
    
    # Train
    trainer.train()
    
    # Save model
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model()
    
    return model, tokenizer


def save_merged_model(
    model,
    tokenizer,
    output_dir: str,
    save_method: str = "merged_16bit",
):
    """Save merged model (LoRA weights merged into base model)."""
    
    print(f"\nSaving merged model to {output_dir}...")
    
    if save_method == "merged_16bit":
        model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")
    elif save_method == "merged_4bit":
        model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_4bit")
    elif save_method == "lora":
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    else:
        raise ValueError(f"Unknown save method: {save_method}")
    
    print(f"Model saved to {output_dir}")


def save_to_gguf(
    model,
    tokenizer,
    output_path: str,
    quantization: str = "q4_k_m",
):
    """Save model in GGUF format."""
    
    print(f"\nSaving GGUF to {output_path}...")
    
    # Unsloth's built-in GGUF export
    model.save_pretrained_gguf(
        output_path,
        tokenizer,
        quantization_method=quantization,
    )
    
    print(f"GGUF saved to {output_path}")


def create_config_template(output_path: str):
    """Create a sample configuration file."""
    
    config = {
        "model": {
            "name": "Qwen/Qwen3-4B-Instruct",
            "max_seq_length": 512,
            "load_in_4bit": True,
        },
        "lora": {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
        },
        "training": {
            "num_train_epochs": 3,
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 1e-4,
            "warmup_steps": 100,
            "save_steps": 500,
            "logging_steps": 10,
        },
        "data": {
            "train_dataset": "data/train.jsonl",
        },
        "output": {
            "output_dir": "outputs/model",
            "save_merged": True,
            "save_gguf": True,
            "gguf_quantization": "q4_k_m",
        },
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    
    print(f"Config template saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Unsloth Training Script for Qwen3 Fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with command line arguments
  python train_unsloth.py \\
    --dataset train.jsonl \\
    --output-dir outputs/model \\
    --lora-rank 16 \\
    --epochs 3
  
  # Train with config file
  python train_unsloth.py --config config.json
  
  # Generate config template
  python train_unsloth.py --create-config config.json
  
  # Train and export to GGUF
  python train_unsloth.py \\
    --dataset train.jsonl \\
    --output-dir outputs/model \\
    --save-gguf \\
    --gguf-quant q4_k_m
        """
    )
    
    # Config
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    parser.add_argument("--create-config", type=str, help="Create config template at path")
    
    # Model
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct",
                        help="Model name or path")
    parser.add_argument("--max-seq-length", type=int, default=512,
                        help="Maximum sequence length")
    
    # Data
    parser.add_argument("--dataset", type=str, help="Path to training dataset (JSONL)")
    
    # LoRA
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    
    # Training
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gradient-accumulation", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--save-steps", type=int, default=500, help="Save every N steps")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="outputs/model",
                        help="Output directory")
    parser.add_argument("--save-merged", action="store_true",
                        help="Save merged model (LoRA + base)")
    parser.add_argument("--save-gguf", action="store_true",
                        help="Export to GGUF format")
    parser.add_argument("--gguf-quant", type=str, default="q4_k_m",
                        choices=["q4_k_m", "q5_k_m", "q8_0", "f16"],
                        help="GGUF quantization method")
    
    # Advanced
    parser.add_argument("--no-4bit", action="store_true",
                        help="Disable 4-bit quantization")
    parser.add_argument("--no-gradient-checkpoint", action="store_true",
                        help="Disable gradient checkpointing")
    parser.add_argument("--skip-checks", action="store_true",
                        help="Skip dependency checks")
    
    args = parser.parse_args()
    
    # Create config template
    if args.create_config:
        create_config_template(args.create_config)
        return
    
    # Check dependencies
    if not args.skip_checks:
        if not check_dependencies():
            sys.exit(1)
    
    # Load config
    config = {}
    if args.config:
        config = load_config(args.config)
    
    # Get parameters (config overrides defaults, args override config)
    model_name = args.model
    dataset_path = args.dataset
    output_dir = args.output_dir
    
    if config:
        model_config = config.get("model", {})
        data_config = config.get("data", {})
        output_config = config.get("output", {})
        
        model_name = model_config.get("name", model_name)
        dataset_path = data_config.get("train_dataset", dataset_path)
        output_dir = output_config.get("output_dir", output_dir)
    
    # Validate required arguments
    if not dataset_path:
        parser.error("--dataset is required (or specify in config file)")
    
    # Build training parameters
    train_params = {
        "model_name": model_name,
        "dataset_path": dataset_path,
        "output_dir": output_dir,
        "max_seq_length": args.max_seq_length,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "num_train_epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "save_steps": args.save_steps,
        "load_in_4bit": not args.no_4bit,
        "use_gradient_checkpointing": not args.no_gradient_checkpoint,
    }
    
    # Override with config values
    if config:
        model_config = config.get("model", {})
        lora_config = config.get("lora", {})
        training_config = config.get("training", {})
        
        train_params.update({
            "max_seq_length": model_config.get("max_seq_length", train_params["max_seq_length"]),
            "load_in_4bit": model_config.get("load_in_4bit", train_params["load_in_4bit"]),
            "lora_rank": lora_config.get("rank", train_params["lora_rank"]),
            "lora_alpha": lora_config.get("alpha", train_params["lora_alpha"]),
            "lora_dropout": lora_config.get("dropout", train_params["lora_dropout"]),
            "num_train_epochs": training_config.get("num_train_epochs", train_params["num_train_epochs"]),
            "batch_size": training_config.get("batch_size", train_params["batch_size"]),
            "gradient_accumulation_steps": training_config.get("gradient_accumulation_steps", train_params["gradient_accumulation_steps"]),
            "learning_rate": training_config.get("learning_rate", train_params["learning_rate"]),
            "warmup_steps": training_config.get("warmup_steps", train_params["warmup_steps"]),
            "save_steps": training_config.get("save_steps", train_params["save_steps"]),
        })
    
    # Train model
    model, tokenizer = train_model(**train_params)
    
    # Save merged model
    save_merged = args.save_merged
    if config:
        save_merged = config.get("output", {}).get("save_merged", save_merged)
    
    if save_merged:
        merged_dir = str(Path(output_dir) / "merged")
        save_merged_model(model, tokenizer, merged_dir)
    
    # Export to GGUF
    save_gguf = args.save_gguf
    gguf_quant = args.gguf_quant
    if config:
        output_config = config.get("output", {})
        save_gguf = output_config.get("save_gguf", save_gguf)
        gguf_quant = output_config.get("gguf_quantization", gguf_quant)
    
    if save_gguf:
        gguf_path = str(Path(output_dir) / f"model-{gguf_quant}.gguf")
        try:
            save_to_gguf(model, tokenizer, gguf_path, gguf_quant)
        except Exception as e:
            print(f"Warning: GGUF export failed: {e}")
            print("You can manually convert using llama.cpp's convert script.")
    
    print(f"\n{'='*60}")
    print("Training completed successfully!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    if save_merged:
        print(f"Merged model: {Path(output_dir) / 'merged'}")
    if save_gguf:
        print(f"GGUF model: {Path(output_dir) / f'model-{gguf_quant}.gguf'}")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("1. Evaluate: python evaluate_gguf.py --model outputs/model/model-q4_k_m.gguf")
    print("2. Deploy to your application")


if __name__ == "__main__":
    main()

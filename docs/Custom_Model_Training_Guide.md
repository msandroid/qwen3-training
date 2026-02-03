# Custom Model Training & Evaluation Guide

A comprehensive guide for building and evaluating custom translation models based on Qwen3. This guide covers Windows, macOS, and Linux environments.

## Table of Contents

1. [Overview](#1-overview)
2. [Environment Setup](#2-environment-setup)
3. [Data Preparation](#3-data-preparation)
4. [Training Methods](#4-training-methods)
5. [Model Evaluation](#5-model-evaluation)
6. [Model Export & Conversion](#6-model-export--conversion)
7. [Troubleshooting](#7-troubleshooting)
8. [References](#8-references)

---

## 1. Overview

### 1.1 Purpose

This guide explains how to fine-tune Qwen3 models for custom translation tasks, with a focus on low-resource languages. The trained models can be deployed on:

- **iOS/macOS**: Using MLX or llama.cpp
- **Android**: Using llama.cpp
- **Windows/Linux**: Using llama.cpp or Transformers

### 1.2 Supported Frameworks

| Framework | Platform | GPU Required | Best For |
|-----------|----------|--------------|----------|
| **MS-SWIFT** | Windows/Linux | CUDA (NVIDIA) | Production training |
| **Unsloth** | Windows/Linux | CUDA (NVIDIA) | Fast training, low VRAM |
| **MLX** | macOS (Apple Silicon) | Apple GPU | Mac-native training |

### 1.3 Hardware Requirements

| Model Size | Minimum VRAM | Recommended |
|------------|--------------|-------------|
| 0.6B-4B | 8-16GB | RTX 3090/4090, T4 |
| 8B | 22GB | A10, RTX 4090 |
| 14B | 24-32GB | A100 40GB |
| 32B | 80GB+ | A100 80GB x2 |

---

## 2. Environment Setup

### 2.1 Windows Setup

#### Option A: WSL2 + CUDA (Recommended)

```powershell
# 1. Enable WSL2
wsl --install -d Ubuntu-22.04

# 2. Install NVIDIA drivers (Windows side)
# Download from: https://www.nvidia.com/drivers

# 3. Inside WSL2, install CUDA toolkit
sudo apt update
sudo apt install -y nvidia-cuda-toolkit

# 4. Verify CUDA
nvidia-smi
```

See [Windows_Setup_Guide.md](Windows_Setup_Guide.md) for detailed instructions.

#### Option B: Native Windows + CUDA

```powershell
# Install Python 3.10+
winget install Python.Python.3.11

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2.2 macOS Setup (Apple Silicon)

```bash
# Install Homebrew if needed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.11

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install MLX
pip install mlx mlx-lm
```

### 2.3 Linux Setup

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2.4 Framework Installation

#### MS-SWIFT (Windows/Linux)

```bash
pip install ms-swift -U
pip install transformers deepspeed
pip install flash-attn --no-build-isolation  # Optional, requires CUDA
```

#### Unsloth (Windows/Linux)

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers trl peft accelerate bitsandbytes
```

#### MLX (macOS only)

```bash
pip install mlx mlx-lm transformers datasets
```

---

## 3. Data Preparation

### 3.1 Data Format

Training data should be in JSONL format with the following structure:

```json
{"messages": [
  {"role": "system", "content": "You are a translation assistant. Output only the translation without explanation."},
  {"role": "user", "content": "Translate from English to Swahili:\n\nHello, how are you?"},
  {"role": "assistant", "content": "Habari, habari gani?"}
]}
```

### 3.2 Key Points

- **System prompt**: Essential for controlling model behavior
- **User message**: Include explicit source/target language instruction
- **Assistant response**: Contains only the translation (no explanations)

### 3.3 Data Sources for Translation

| Dataset | Languages | Description | License |
|---------|-----------|-------------|---------|
| [MAFAND-MT](https://github.com/masakhane-io/masakhane-mt) | 21 | High-quality African language pairs | CC-BY-4.0 |
| [JW300](https://opus.nlpl.eu/JW300.php) | 300+ | Religious texts, broad coverage | CC-BY-NC-SA |
| [FLORES-200](https://github.com/facebookresearch/flores) | 200 | Evaluation benchmark | CC-BY-SA-4.0 |
| [Tatoeba](https://tatoeba.org/) | 400+ | Community-contributed sentences | CC-BY-2.0 |

### 3.4 Data Preparation Script

```python
import json
from pathlib import Path

def prepare_translation_data(
    input_file: str,
    output_file: str,
    source_lang: str,
    target_lang: str,
    system_prompt: str = "You are a translation assistant. Output only the translation."
):
    """Convert raw translation pairs to training format."""
    
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    processed = []
    for item in raw_data:
        source_text = item.get('source', '')
        target_text = item.get('target', '')
        
        if not source_text or not target_text:
            continue
        
        entry = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Translate from {source_lang} to {target_lang}:\n\n{source_text}"},
                {"role": "assistant", "content": target_text}
            ]
        }
        processed.append(entry)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in processed:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Processed {len(processed)} samples -> {output_file}")

# Example usage
# prepare_translation_data('raw_en_sw.json', 'train.jsonl', 'English', 'Swahili')
```

### 3.5 Train/Validation/Test Split

```python
import random
from pathlib import Path

def split_data(input_file: str, output_dir: str, train_ratio: float = 0.9, val_ratio: float = 0.05):
    """Split data into train/validation/test sets."""
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    random.shuffle(lines)
    
    n = len(lines)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / 'train.jsonl', 'w') as f:
        f.writelines(lines[:train_end])
    
    with open(output_path / 'valid.jsonl', 'w') as f:
        f.writelines(lines[train_end:val_end])
    
    with open(output_path / 'test.jsonl', 'w') as f:
        f.writelines(lines[val_end:])
    
    print(f"Split: train={train_end}, valid={val_end-train_end}, test={n-val_end}")
```

---

## 4. Training Methods

### 4.1 MS-SWIFT (Recommended for Windows/Linux)

#### Basic LoRA Training

```bash
CUDA_VISIBLE_DEVICES=0 swift sft \
  --model Qwen/Qwen3-4B-Instruct \
  --tuner_type lora \
  --dataset path/to/train.jsonl \
  --val_dataset path/to/valid.jsonl \
  --torch_dtype bfloat16 \
  --learning_rate 1e-4 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --max_length 512 \
  --num_train_epochs 3 \
  --output_dir outputs/my-translation-model
```

#### Key Parameters

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `lora_rank` | 16-32 | Higher = more capacity, more VRAM |
| `lora_alpha` | 32-64 | Scaling factor (usually 2x rank) |
| `learning_rate` | 1e-4 to 5e-5 | Lower for larger models |
| `max_length` | 512-1024 | Sequence length limit |
| `num_train_epochs` | 2-5 | More epochs for small datasets |

### 4.2 Unsloth (Fast, Low VRAM)

```python
from unsloth import FastLanguageModel
import torch

# Load model with 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-4B-Instruct",
    max_seq_length=512,
    dtype=None,  # Auto-detect
    load_in_4bit=True,
)

# Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# Training with HuggingFace Trainer
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

dataset = load_dataset("json", data_files="train.jsonl", split="train")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        num_train_epochs=3,
        learning_rate=1e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        output_dir="outputs",
        save_steps=500,
    ),
)

trainer.train()
```

### 4.3 MLX (macOS Apple Silicon)

```bash
# Using MLX-LM CLI
python -m mlx_lm.lora \
  --model mlx-community/Qwen3-4B-Instruct-4bit \
  --train \
  --data data/processed \
  --adapter-path outputs/lora-adapter \
  --batch-size 2 \
  --lora-layers 16 \
  --learning-rate 1e-4 \
  --iters 5000 \
  --max-seq-length 512 \
  --save-every 500
```

### 4.4 Training Tips

1. **Start small**: Test with 1000 samples first
2. **Monitor loss**: Validation loss should decrease steadily
3. **Save checkpoints**: Save every 500-1000 steps
4. **Use system prompts**: Critical for translation quality
5. **Avoid overfitting**: Stop if validation loss increases

---

## 5. Model Evaluation

### 5.1 Evaluation Metrics

| Metric | Range | Description |
|--------|-------|-------------|
| **BLEU** | 0-100 | N-gram overlap (higher = better) |
| **chrF** | 0-100 | Character n-gram F-score |
| **TER** | 0-∞ | Translation Edit Rate (lower = better) |
| **COMET** | 0-1 | Neural quality metric |

### 5.2 Target Scores

| Quality Level | BLEU | chrF | TER |
|---------------|------|------|-----|
| Minimum Viable | ≥15 | ≥40 | <100 |
| Production Ready | ≥20 | ≥45 | <80 |
| High Quality | ≥25 | ≥50 | <70 |

### 5.3 Evaluation Script

See [examples/scripts/evaluate_gguf.py](../examples/scripts/evaluate_gguf.py) for a complete evaluation script.

```python
import sacrebleu

def evaluate_translations(hypotheses: list[str], references: list[str]) -> dict:
    """Compute BLEU, chrF, and TER scores."""
    
    refs = [references]  # sacrebleu expects list of reference lists
    
    bleu = sacrebleu.corpus_bleu(hypotheses, refs)
    chrf = sacrebleu.corpus_chrf(hypotheses, refs)
    ter = sacrebleu.corpus_ter(hypotheses, refs)
    
    return {
        "bleu": round(bleu.score, 2),
        "chrf": round(chrf.score, 2),
        "ter": round(ter.score, 2),
    }

# Example usage
hypotheses = ["Habari, habari gani?", "Asante sana"]
references = ["Habari, unaendeleaje?", "Asante sana"]
scores = evaluate_translations(hypotheses, references)
print(scores)  # {'bleu': 45.2, 'chrf': 62.1, 'ter': 35.8}
```

### 5.4 Running Evaluation with GGUF Model

```bash
# Install dependencies
pip install llama-cpp-python sacrebleu

# Run evaluation
python examples/scripts/evaluate_gguf.py \
  --model path/to/model.gguf \
  --test-data path/to/test.jsonl \
  --max-samples 100 \
  --output-dir results/
```

---

## 6. Model Export & Conversion

### 6.1 Export Formats

| Format | Platform | Tool |
|--------|----------|------|
| **GGUF** | Cross-platform | llama.cpp |
| **MLX** | macOS/iOS | mlx-lm |
| **CoreML** | iOS | coremltools |
| **SafeTensors** | All | HuggingFace |

### 6.2 Merge LoRA Adapter

#### MS-SWIFT

```bash
swift export \
  --model Qwen/Qwen3-4B-Instruct \
  --adapter outputs/my-translation-model \
  --output_dir outputs/merged-model \
  --merge_lora true
```

#### Unsloth

```python
# Save merged model
model.save_pretrained_merged("outputs/merged-model", tokenizer)
```

#### MLX

```bash
python -m mlx_lm.fuse \
  --model mlx-community/Qwen3-4B-Instruct-4bit \
  --adapter-path outputs/lora-adapter \
  --save-path outputs/merged-model
```

### 6.3 Convert to GGUF

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Install dependencies
pip install -r requirements.txt

# Convert to GGUF (F16)
python convert_hf_to_gguf.py \
  ../outputs/merged-model \
  --outfile ../outputs/model-f16.gguf \
  --outtype f16

# Quantize to Q4_K_M (recommended for mobile)
./llama-quantize \
  ../outputs/model-f16.gguf \
  ../outputs/model-q4_k_m.gguf \
  Q4_K_M
```

### 6.4 Quantization Options

| Type | Size | Quality | Use Case |
|------|------|---------|----------|
| Q2_K | ~1.5GB | Lower | Extreme memory constraints |
| Q4_K_M | ~2.3GB | Good | **Recommended for mobile** |
| Q5_K_M | ~2.8GB | Better | Desktop with limited VRAM |
| Q8_0 | ~4.3GB | Best | Desktop with ample VRAM |
| F16 | ~8GB | Original | No quantization |

---

## 7. Troubleshooting

### 7.1 Out of Memory (OOM)

**Symptoms:**
- Training crashes with "CUDA out of memory"
- System becomes unresponsive

**Solutions:**
1. Reduce batch size: `--batch_size 1`
2. Reduce sequence length: `--max_length 256`
3. Enable gradient checkpointing
4. Use 4-bit quantization (QLoRA)
5. Reduce LoRA rank: `--lora_rank 8`

### 7.2 Repetition Loops

**Symptoms:**
- Model generates "word word word..." repeatedly
- Output gets stuck in cycles

**Solutions:**
1. Add system prompt enforcing direct translation
2. Increase LoRA rank to 16-32
3. Lower temperature: `--temperature 0.3`
4. Set repetition penalty: `--repeat_penalty 1.1`
5. Use top-k sampling: `--top_k 20`

### 7.3 Poor Translation Quality

**Symptoms:**
- BLEU score < 10
- Hallucinated content
- Wrong target language

**Solutions:**
1. Verify data format (system prompts present)
2. Increase training iterations
3. Use higher LoRA rank (16-32)
4. Check for data quality issues
5. Try different base model

### 7.4 GGUF Export Fails

**Error:** `Unsupported model_type: qwen3`

**Solution:** Use llama.cpp's convert script directly:

```bash
# Use llama.cpp converter
python llama.cpp/convert_hf_to_gguf.py \
  outputs/merged-model \
  --outfile outputs/model-f16.gguf \
  --outtype f16
```

### 7.5 Flash Attention Installation Fails (Windows)

**Solution:** Skip flash-attn or use WSL2:

```bash
# Option 1: Skip flash-attn (slower but works)
pip install ms-swift -U
# Don't install flash-attn

# Option 2: Use WSL2 (recommended)
# Install in Ubuntu WSL2 environment
```

---

## 8. References

### 8.1 Official Documentation

- [Qwen3 GitHub](https://github.com/QwenLM/Qwen3)
- [MS-SWIFT Documentation](https://swift.readthedocs.io/)
- [Unsloth Documentation](https://unsloth.ai/docs)
- [MLX Examples](https://github.com/ml-explore/mlx-examples)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)

### 8.2 Tutorials

- [DataCamp: Fine-tuning Qwen3](https://www.datacamp.com/tutorial/fine-tuning-qwen3)
- [Qwen3 Best Practices](https://swift.readthedocs.io/en/latest/BestPractices/Qwen3-Best-Practice.html)

### 8.3 Datasets

- [OPUS Parallel Corpora](https://opus.nlpl.eu/)
- [Masakhane African NLP](https://www.masakhane.io/)
- [Hugging Face Datasets](https://huggingface.co/datasets)

### 8.4 Evaluation Tools

- [sacrebleu](https://github.com/mjpost/sacrebleu)
- [COMET](https://github.com/Unbabel/COMET)

---

## Appendix: Quick Reference

### Training Command Templates

**MS-SWIFT (Windows/Linux):**
```bash
swift sft --model Qwen/Qwen3-4B-Instruct --tuner_type lora --dataset train.jsonl --lora_rank 16 --learning_rate 1e-4
```

**MLX (macOS):**
```bash
python -m mlx_lm.lora --model mlx-community/Qwen3-4B-Instruct-4bit --train --data data/ --iters 5000
```

### Evaluation Command

```bash
python evaluate_gguf.py --model model.gguf --test-data test.jsonl --max-samples 100
```

### GGUF Conversion

```bash
python convert_hf_to_gguf.py model/ --outfile model.gguf --outtype f16
./llama-quantize model.gguf model-q4.gguf Q4_K_M
```

---

*Last Updated: February 2026*

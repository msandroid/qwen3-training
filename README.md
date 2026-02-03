# Qwen3 Training

Qwen3 model fine-tuning and training resources. Documentation and examples for SFT, LoRA, and RLHF workflows.

**Supports:** Windows (WSL2/Native), macOS (Apple Silicon), Linux

## Contents

- [docs/](docs/) - Comprehensive guides and documentation
  - [Custom Model Training Guide](docs/Custom_Model_Training_Guide.md) - Complete training & evaluation workflow
  - [Windows Setup Guide](docs/Windows_Setup_Guide.md) - Windows-specific setup instructions
  - [Qwen3 Fine-tuning Guide](docs/Qwen3_Fine-tuning_Guide.md) - Framework reference (Japanese)
- [examples/](examples/) - Sample datasets and training scripts
  - [scripts/](examples/scripts/) - Training and evaluation scripts
  - [data/](examples/data/) - Sample datasets

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (NVIDIA) or Apple Silicon (M1/M2/M3)
- 16GB+ VRAM for small models (0.6B-4B)

### Installation

#### Windows (WSL2 Recommended)

```bash
# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install MS-SWIFT
pip install ms-swift -U

# Install evaluation tools
pip install llama-cpp-python sacrebleu
```

See [Windows Setup Guide](docs/Windows_Setup_Guide.md) for detailed instructions.

#### macOS (Apple Silicon)

```bash
# Install MLX
pip install mlx mlx-lm

# Install evaluation tools
pip install llama-cpp-python sacrebleu
```

#### Linux

```bash
# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install MS-SWIFT or Unsloth
pip install ms-swift -U
# OR
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### Training with MS-SWIFT

```bash
CUDA_VISIBLE_DEVICES=0 swift sft \
  --model Qwen/Qwen3-4B-Instruct \
  --tuner_type lora \
  --dataset examples/data/sample_translation.jsonl \
  --torch_dtype bfloat16 \
  --learning_rate 1e-4 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --max_length 512
```

### Training with Python Scripts

```bash
# MS-SWIFT wrapper
python examples/scripts/train_msswift.py \
  --dataset examples/data/sample_translation.jsonl \
  --output-dir outputs/model \
  --lora-rank 16 \
  --epochs 3

# Unsloth (faster, lower VRAM)
python examples/scripts/train_unsloth.py \
  --dataset examples/data/sample_translation.jsonl \
  --output-dir outputs/model \
  --save-gguf
```

### Evaluation

```bash
python examples/scripts/evaluate_gguf.py \
  --model outputs/model/model-q4_k_m.gguf \
  --test-data examples/data/sample_translation.jsonl \
  --max-samples 100
```

## Documentation

| Document | Description |
|----------|-------------|
| [Custom Model Training Guide](docs/Custom_Model_Training_Guide.md) | Complete guide for building and evaluating custom models |
| [Windows Setup Guide](docs/Windows_Setup_Guide.md) | WSL2 and native Windows setup instructions |
| [Qwen3 Fine-tuning Guide](docs/Qwen3_Fine-tuning_Guide.md) | Framework reference and best practices |

## Scripts

| Script | Description |
|--------|-------------|
| [train_msswift.py](examples/scripts/train_msswift.py) | MS-SWIFT training wrapper |
| [train_unsloth.py](examples/scripts/train_unsloth.py) | Unsloth training (2x faster, 70% less VRAM) |
| [evaluate_gguf.py](examples/scripts/evaluate_gguf.py) | GGUF model evaluation (BLEU/chrF/TER) |

## Sample Data

| File | Description |
|------|-------------|
| [sample_sft.jsonl](examples/data/sample_sft.jsonl) | General SFT examples |
| [sample_translation.jsonl](examples/data/sample_translation.jsonl) | Translation task examples (multiple languages) |

## Supported Frameworks

| Framework | Platform | Best For |
|-----------|----------|----------|
| **MS-SWIFT** | Windows/Linux | Production training, full features |
| **Unsloth** | Windows/Linux | Fast training, low VRAM |
| **MLX** | macOS (Apple Silicon) | Native Mac training |

## Hardware Requirements

| Model Size | Minimum VRAM | Recommended GPU |
|------------|--------------|-----------------|
| 0.6B-4B | 8-16GB | RTX 3090/4090, T4 |
| 8B | 22GB | A10, RTX 4090 |
| 14B | 24-32GB | A100 40GB |
| 32B | 80GB+ | A100 80GB x2 |

## Resources

- [Qwen3 Official](https://github.com/QwenLM/Qwen3)
- [MS-SWIFT](https://github.com/modelscope/ms-swift)
- [Unsloth](https://github.com/unslothai/unsloth)
- [MLX Examples](https://github.com/ml-explore/mlx-examples)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)

## License

Documentation and examples are provided for reference. Qwen3 model usage follows the [Qwen License](https://github.com/QwenLM/Qwen3/blob/main/LICENSE).

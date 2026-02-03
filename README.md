# Qwen3 Training

Qwen3 model fine-tuning and training resources. Documentation and examples for SFT, LoRA, and RLHF workflows.

## Contents

- [docs/](docs/) - Fine-tuning guide and reference
- [examples/](examples/) - Sample datasets and training scripts

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (see [Hardware Requirements](docs/Qwen3_Fine-tuning_Guide.md#5-ハードウェア要件))
- 16GB+ VRAM for small models (0.6B-4B)

### Installation (MS-SWIFT)

```bash
pip install ms-swift -U
pip install transformers deepspeed liger-kernel
pip install flash-attn --no-build-isolation
```

### SFT with LoRA

```bash
CUDA_VISIBLE_DEVICES=0 swift sft \
  --model Qwen/Qwen3-8B \
  --tuner_type lora \
  --dataset examples/data/sample_sft.jsonl \
  --torch_dtype bfloat16 \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --max_length 2048
```

## Resources

- [Qwen3 Fine-tuning Guide](docs/Qwen3_Fine-tuning_Guide.md)
- [Qwen3 Official](https://github.com/QwenLM/Qwen3)
- [MS-SWIFT](https://github.com/modelscope/ms-swift)

## License

Documentation and examples are provided for reference. Qwen3 model usage follows the [Qwen License](https://github.com/QwenLM/Qwen3/blob/main/LICENSE).

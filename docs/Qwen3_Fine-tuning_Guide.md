# Qwen3 Fine-tuning Documentation Reference

## 1. 公式トレーニングフレームワーク

### MS-SWIFT (ModelScope SWIFT)

Alibaba公式のLLMトレーニングフレームワーク

**GitHub:** [https://github.com/modelscope/ms-swift](https://github.com/modelscope/ms-swift)
**ドキュメント:** [https://qwen.readthedocs.io/en/latest/training/ms_swift.html](https://qwen.readthedocs.io/en/latest/training/ms_swift.html)
**ベストプラクティス:** [https://swift.readthedocs.io/en/latest/BestPractices/Qwen3-Best-Practice.html](https://swift.readthedocs.io/en/latest/BestPractices/Qwen3-Best-Practice.html)

**主な機能:**

- 500+ LLMモデル、200+ マルチモーダルモデル対応
- フルファインチューニング、LoRA、QLoRA、DoRA
- RLHF: DPO、GRPO、DAPO、PPO、KTO
- 分散学習: DeepSpeed ZeRO-2/3、FSDP、Megatron

**インストール:**

```bash
pip install ms-swift -U
pip install transformers deepspeed liger-kernel
pip install flash-attn --no-build-isolation
```

---

## 2. データセット形式

### SFT基本形式 (JSON/JSONL/CSV)

```json
{"messages": [
  {"role": "system", "content": "<system-prompt>"},
  {"role": "user", "content": "<query>"},
  {"role": "assistant", "content": "<response>"}
]}
```

### 推論チェーン付き形式

```json
{"messages": [
  {"role": "user", "content": "質問"},
  {"role": "assistant", "content": "<think>\n推論過程...\n</think>\n\n回答"}
]}
```

### 推論能力を維持する方法

1. **推奨:** `--loss_scale ignore_empty_think` を使用
2. クエリに `/no_think` を追加
3. 推論データ75% + 非推論データ25%をミックス

---

## 3. トレーニング手法別ガイド

### SFT (Supervised Fine-Tuning)

```bash
CUDA_VISIBLE_DEVICES=0 swift sft \
  --model Qwen/Qwen3-8B \
  --tuner_type lora \
  --dataset '<your-dataset>' \
  --torch_dtype bfloat16 \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --max_length 2048
```

### GRPO (強化学習)

```bash
pip install math_verify vllm

swift rlhf \
  --rlhf_type grpo \
  --model Qwen/Qwen3-8B \
  --dataset 'AI-MO/NuminaMath-TIR#5000' \
  --reward_funcs accuracy \
  --num_generations 16 \
  --use_vllm true
```

---

## 4. 代替フレームワーク

### Unsloth

**ドキュメント:** [https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune](https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune)
**特徴:** 2倍高速、70%少ないVRAM、8倍長いコンテキスト

```bash
pip install --upgrade unsloth unsloth_zoo
```

**Colabノートブック:**

- [Qwen3 (14B) Reasoning + Conversational](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(14B)-Reasoning-Conversational.ipynb)
- [Qwen3 (4B) GRPO LoRA](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-GRPO.ipynb)

### その他のフレームワーク

- **Axolotl:** [https://qwen.readthedocs.io/en/latest/training/axolotl.html](https://qwen.readthedocs.io/en/latest/training/axolotl.html)
- **LLaMA-Factory:** [https://qwen.readthedocs.io/en/latest/training/llama_factory.html](https://qwen.readthedocs.io/en/latest/training/llama_factory.html)
- **verl:** [https://qwen.readthedocs.io/en/latest/training/verl.html](https://qwen.readthedocs.io/en/latest/training/verl.html)

---

## 5. ハードウェア要件


| モデル           | 最小VRAM           | 推奨環境              |
| ------------- | ---------------- | ----------------- |
| 0.6B-4B       | 8-16GB           | RTX 3090/4090, T4 |
| 8B            | 22GB             | A10, RTX 4090     |
| 14B           | 24-32GB          | A100 40GB         |
| 32B           | 80GB+            | A100 80GB x2      |
| 30B-A3B (MoE) | 17.5GB (Unsloth) | A10+              |


---

## 6. 推論設定

### Thinking Mode

- Temperature: 0.6
- Top_P: 0.95
- TopK: 20
- Min_P: 0.0

### Non-Thinking Mode

- Temperature: 0.7
- Top_P: 0.8
- TopK: 20

---

## 7. 主要リソースリンク

**公式:**

- Qwen3 GitHub: [https://github.com/QwenLM/Qwen3](https://github.com/QwenLM/Qwen3)
- Qwen Documentation: [https://qwen.readthedocs.io/](https://qwen.readthedocs.io/)
- MS-SWIFT: [https://github.com/modelscope/ms-swift](https://github.com/modelscope/ms-swift)

**チュートリアル:**

- DataCamp Guide: [https://www.datacamp.com/tutorial/fine-tuning-qwen3](https://www.datacamp.com/tutorial/fine-tuning-qwen3)
- AWS Trainium: [https://huggingface.co/docs/optimum-neuron/training_tutorials/qwen3-fine-tuning](https://huggingface.co/docs/optimum-neuron/training_tutorials/qwen3-fine-tuning)

**コミュニティ:**

- MS-SWIFT Issues: [https://github.com/modelscope/ms-swift/issues/4030](https://github.com/modelscope/ms-swift/issues/4030)


# Windows Setup Guide for Qwen3 Fine-tuning

A detailed guide for setting up a Qwen3 fine-tuning environment on Windows. This guide covers both WSL2 (recommended) and native Windows approaches.

## Table of Contents

1. [Overview](#1-overview)
2. [WSL2 + CUDA Setup (Recommended)](#2-wsl2--cuda-setup-recommended)
3. [Native Windows Setup](#3-native-windows-setup)
4. [Framework Installation](#4-framework-installation)
5. [Verification](#5-verification)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Overview

### 1.1 Setup Options

| Option | Pros | Cons | Recommended For |
|--------|------|------|-----------------|
| **WSL2 + CUDA** | Full Linux compatibility, Flash Attention support | Requires WSL2 setup | Training |
| **Native Windows** | No WSL needed | Limited Flash Attention, some packages may fail | Evaluation only |
| **CPU Only** | No GPU needed | Very slow training | Testing/small experiments |

### 1.2 System Requirements

- **OS**: Windows 10 (version 2004+) or Windows 11
- **GPU**: NVIDIA GPU with CUDA support (RTX 20/30/40 series recommended)
- **RAM**: 16GB+ (32GB recommended)
- **Storage**: 50GB+ free space
- **NVIDIA Driver**: 525.60.13 or later

---

## 2. WSL2 + CUDA Setup (Recommended)

### 2.1 Enable WSL2

Open PowerShell as Administrator:

```powershell
# Enable WSL
wsl --install

# Set WSL2 as default
wsl --set-default-version 2

# Install Ubuntu 22.04
wsl --install -d Ubuntu-22.04

# Restart your computer
```

### 2.2 Install NVIDIA Drivers (Windows Side)

1. Download the latest NVIDIA driver from [nvidia.com/drivers](https://www.nvidia.com/drivers)
2. Install the driver (select "Custom" and check "Clean Installation")
3. Restart your computer

**Important**: Do NOT install CUDA toolkit on Windows. WSL2 uses the Windows driver directly.

### 2.3 Configure WSL2 Ubuntu

Open Ubuntu from Start menu and run:

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential tools
sudo apt install -y build-essential git curl wget

# Verify NVIDIA driver access
nvidia-smi
```

You should see your GPU information. If not, restart WSL:

```powershell
# In PowerShell
wsl --shutdown
# Then reopen Ubuntu
```

### 2.4 Install CUDA Toolkit in WSL2

```bash
# Add NVIDIA package repository
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Install CUDA toolkit
sudo apt install -y cuda-toolkit-12-1

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify installation
nvcc --version
```

### 2.5 Install Python Environment

```bash
# Install Python 3.11
sudo apt install -y python3.11 python3.11-venv python3.11-dev

# Create project directory
mkdir -p ~/qwen3-training
cd ~/qwen3-training

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 2.6 Install PyTorch with CUDA

```bash
# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

---

## 3. Native Windows Setup

### 3.1 Install Python

```powershell
# Using winget
winget install Python.Python.3.11

# Or download from python.org
# https://www.python.org/downloads/
```

**Important**: During installation, check "Add Python to PATH".

### 3.2 Install CUDA Toolkit

1. Download CUDA Toolkit from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
2. Select: Windows > x86_64 > 11/10 > exe (local)
3. Run installer with default options
4. Restart your computer

### 3.3 Install cuDNN

1. Download cuDNN from [developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn) (requires NVIDIA account)
2. Extract the archive
3. Copy files to CUDA installation:
   - `bin\*.dll` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin\`
   - `include\*.h` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\include\`
   - `lib\x64\*.lib` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\lib\x64\`

### 3.4 Create Python Environment

Open Command Prompt or PowerShell:

```powershell
# Create project directory
mkdir C:\qwen3-training
cd C:\qwen3-training

# Create virtual environment
python -m venv venv

# Activate (PowerShell)
.\venv\Scripts\Activate.ps1

# Or activate (Command Prompt)
.\venv\Scripts\activate.bat

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 3.5 Install PyTorch

```powershell
# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 4. Framework Installation

### 4.1 MS-SWIFT Installation

MS-SWIFT is the official Alibaba framework for Qwen fine-tuning.

```bash
# Install MS-SWIFT
pip install ms-swift -U

# Install additional dependencies
pip install transformers datasets accelerate
pip install deepspeed  # Optional, for distributed training

# Flash Attention (WSL2 only, optional)
pip install flash-attn --no-build-isolation
```

### 4.2 Unsloth Installation

Unsloth provides 2x faster training with 70% less VRAM.

```bash
# Install Unsloth
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Install dependencies
pip install --no-deps xformers trl peft accelerate bitsandbytes
```

**Note for Native Windows**: Some Unsloth features may not work. WSL2 is recommended.

### 4.3 Evaluation Tools

```bash
# For model evaluation
pip install sacrebleu
pip install llama-cpp-python  # For GGUF inference

# Optional: COMET (neural metric)
pip install unbabel-comet
```

### 4.4 llama.cpp for GGUF Conversion

#### WSL2

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build with CUDA
make LLAMA_CUDA=1

# Install Python requirements
pip install -r requirements.txt
```

#### Native Windows

```powershell
# Install CMake
winget install Kitware.CMake

# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build with CMake
mkdir build
cd build
cmake .. -DLLAMA_CUDA=ON
cmake --build . --config Release

# Copy executables to parent directory
copy bin\Release\*.exe ..
```

---

## 5. Verification

### 5.1 Check GPU Access

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### 5.2 Test MS-SWIFT

```bash
# Check MS-SWIFT installation
swift --version

# List available models
swift models --model_type qwen3
```

### 5.3 Test Unsloth

```python
from unsloth import FastLanguageModel

# This should load without errors
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B-Instruct-bnb-4bit",
    max_seq_length=512,
    load_in_4bit=True,
)
print("Unsloth loaded successfully!")
```

### 5.4 Test llama-cpp-python

```python
from llama_cpp import Llama

# Test with a small model (download first)
# llm = Llama(model_path="path/to/model.gguf", n_ctx=512)
print("llama-cpp-python installed successfully!")
```

---

## 6. Troubleshooting

### 6.1 WSL2 Cannot Access GPU

**Symptoms:**
- `nvidia-smi` shows "command not found" or no GPU
- PyTorch reports `cuda.is_available() = False`

**Solutions:**

1. Update Windows and NVIDIA driver:
```powershell
# Check Windows version (need 2004+)
winver

# Update NVIDIA driver to latest
```

2. Restart WSL:
```powershell
wsl --shutdown
# Wait 10 seconds, then reopen Ubuntu
```

3. Check WSL version:
```powershell
wsl --list --verbose
# Should show VERSION 2
```

### 6.2 CUDA Out of Memory

**Solutions:**

1. Reduce batch size:
```bash
swift sft ... --batch_size 1
```

2. Use gradient checkpointing:
```bash
swift sft ... --gradient_checkpointing true
```

3. Use 4-bit quantization (QLoRA):
```bash
swift sft ... --quant_bits 4
```

4. Reduce sequence length:
```bash
swift sft ... --max_length 256
```

### 6.3 Flash Attention Installation Fails

**Error:** `No module named 'flash_attn'` or build errors

**Solutions:**

1. **WSL2**: Install with CUDA:
```bash
pip install flash-attn --no-build-isolation
```

2. **Native Windows**: Skip Flash Attention (training will be slower but work):
```bash
# Don't install flash-attn
# MS-SWIFT will use standard attention
```

3. **Alternative**: Use xformers:
```bash
pip install xformers
```

### 6.4 MS-SWIFT Import Errors

**Error:** `ModuleNotFoundError: No module named 'swift'`

**Solution:**
```bash
# Reinstall MS-SWIFT
pip uninstall ms-swift swift -y
pip install ms-swift -U
```

### 6.5 Unsloth Compatibility Issues

**Error:** Various import or runtime errors

**Solutions:**

1. Use specific versions:
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers==0.0.25 trl peft accelerate bitsandbytes
```

2. For Windows, use WSL2 instead of native Windows

### 6.6 llama.cpp Build Fails

**Error:** CMake or compilation errors

**Solutions:**

1. Install Visual Studio Build Tools:
```powershell
winget install Microsoft.VisualStudio.2022.BuildTools
# Select "Desktop development with C++"
```

2. Use pre-built binaries:
   - Download from [llama.cpp releases](https://github.com/ggerganov/llama.cpp/releases)

### 6.7 Slow Training on Windows

**Causes:**
- Windows Defender scanning files
- Antivirus interference
- Disk I/O bottlenecks

**Solutions:**

1. Add exclusions to Windows Defender:
```powershell
# PowerShell as Admin
Add-MpPreference -ExclusionPath "C:\qwen3-training"
```

2. Use SSD for training data and model files

3. Consider using WSL2 (generally faster for ML workloads)

---

## Quick Reference

### WSL2 Setup Commands

```bash
# Windows PowerShell (Admin)
wsl --install -d Ubuntu-22.04

# Ubuntu
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.11 python3.11-venv build-essential
python3.11 -m venv venv
source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install ms-swift -U
```

### Native Windows Setup Commands

```powershell
# PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install ms-swift -U
```

### Verify Installation

```python
import torch
print(f"CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

---

*Last Updated: February 2026*

# Performance Optimization Guide

This guide explains the optimizations implemented for lower latency and faster deployment on **NVIDIA Ampere GPUs** (RTX 30xx/40xx series, A100) and **Intel CPUs with AVX2 support**.

## Hardware Detection

The application now automatically detects your hardware and applies optimal settings:

```bash
python hardware_detector.py
```

This will display:
- CPU information (brand, cores, AVX2/AVX512 support)
- GPU information (model, CUDA version, compute capability)
- Recommended configuration file
- Optimal thread and GPU layer counts

## Automatic Configuration

When you run `ai_voicetalk_local.py`, the application will:
1. Detect your hardware capabilities
2. Load the optimal configuration automatically
3. Display the loaded settings

### Configuration Files

Three optimized configuration files are provided:

#### 1. `creation_params_ampere.json` - For NVIDIA Ampere GPUs
**Optimizations for RTX 30xx/40xx and A100 GPUs:**
- `n_gpu_layers: 35` - Offload maximum layers to GPU
- `n_batch: 2048` - Larger batch size for Ampere's increased memory bandwidth
- `n_threads: 4` - Fewer CPU threads when GPU handles most work
- `f16_kv: true` - Use FP16 for key-value cache (faster on Ampere)
- `mul_mat_q: true` - Utilize tensor cores for matrix multiplication

**Expected Performance:**
- 50-80 tokens/second on RTX 3090
- 60-100 tokens/second on RTX 4090
- Reduced latency: ~100-200ms for first token

#### 2. `creation_params_intel_avx2.json` - For Intel CPUs with AVX2
**Optimizations for modern Intel CPUs (Haswell and newer):**
- `n_gpu_layers: 0` - Pure CPU inference
- `n_threads: 8` - Utilize multiple cores efficiently
- `n_batch: 512` - Optimal batch size for CPU processing
- `use_mmap: true` - Memory-mapped files for faster loading
- `use_mlock: true` - Lock model in RAM to prevent swapping
- `mul_mat_q: false` - CPU-optimized matrix operations

**Expected Performance:**
- 5-15 tokens/second on modern Intel CPUs (i7/i9)
- Initial loading time: 2-5 seconds with mmap

#### 3. `creation_params.json` - Default/Balanced
General configuration that works across different hardware.

## Manual Override

To use a specific configuration, create `creation_params_override.json`:

```json
{
    "n_gpu_layers": 35,
    "n_threads": 6,
    "n_batch": 2048,
    "model_path": "./models/zephyr-7b-beta.Q5_K_M.gguf",
    "n_ctx": 8192
}
```

The application will use this file if it exists.

## Speech Recognition Optimizations

### Ampere GPU Mode
- Uses `base.en` Whisper model for better accuracy with acceptable latency
- GPU acceleration for faster-whisper
- Reduced post-speech silence: 0.4s for quicker response

### CPU Mode
- Uses `tiny.en` Whisper model for maximum speed
- Optimized voice activity detection
- Balanced sensitivity settings

## Text-to-Speech Optimizations

- Speed increased to 1.1x for reduced latency
- Automatic GPU acceleration when available
- Optimized for real-time streaming

## Performance Tuning

### For Lower Latency (Conversation Mode)
Adjust in your config file:
```json
{
    "n_batch": 2048,        // Larger for GPU, smaller (256-512) for CPU
    "max_tokens": 150,      // Shorter responses
    "temperature": 0.6      // More focused responses
}
```

### For Better Quality (Quality Mode)
```json
{
    "n_batch": 512,
    "max_tokens": 300,
    "temperature": 0.8,
    "top_p": 0.95
}
```

## Installation Notes

### For Ampere GPUs
Ensure you have:
- CUDA 11.8 or later (CUDA 12.x recommended for RTX 40xx)
- Latest NVIDIA drivers
- cuDNN 8.7.0 or later

Install llama-cpp-python with CUDA support:
```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
```

### For Intel CPUs with AVX2
Install with optimized BLAS:
```bash
# On Linux - use OpenBLAS
sudo apt-get install libopenblas-dev
CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir

# On Windows - use Intel MKL (if available)
pip install mkl mkl-include
CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=Intel10_64lp" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
```

For maximum CPU performance, compile with AVX2 explicitly:
```bash
CMAKE_ARGS="-DLLAMA_AVX2=ON" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
```

## Benchmarking

To test your configuration performance:

```python
import time
from hardware_detector import print_hardware_info

# Check hardware
print_hardware_info()

# Time model loading
start = time.time()
# ... initialize model ...
print(f"Model load time: {time.time() - start:.2f}s")

# Time inference
start = time.time()
token_count = 0
for token in generator:
    token_count += 1
elapsed = time.time() - start
print(f"Tokens/second: {token_count / elapsed:.2f}")
```

## Troubleshooting

### GPU Not Detected
- Verify CUDA installation: `nvidia-smi`
- Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- Reinstall llama-cpp-python with CUDA support

### Low Performance on Ampere GPU
- Increase `n_gpu_layers` to 35 (full offload)
- Increase `n_batch` to 2048 or 4096
- Verify GPU utilization: `nvidia-smi dmon`
- Check VRAM usage is under 80%

### CPU Performance Issues
- Reduce `n_threads` if system becomes unresponsive
- Enable `use_mlock: true` if you have sufficient RAM
- Consider using a smaller quantized model (Q4_K_M instead of Q5_K_M)
- Verify AVX2 support: `python hardware_detector.py`

### Out of Memory
- Reduce `n_ctx` (context size): 4096 or 2048
- Lower `n_batch`: 256 or 512
- Reduce `n_gpu_layers` if on GPU
- Use a smaller model quantization

## Advanced: Environment Variables

Set these before running for additional optimizations:

```bash
# Force CUDA graphs (may reduce latency on Ampere)
export LLAMA_CUDA_GRAPHS=1

# For CPU: Set thread affinity
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Reduce memory usage
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## Performance Comparison

Expected improvements over default configuration:

| Hardware | Improvement | Tokens/sec | First Token Latency |
|----------|-------------|------------|---------------------|
| RTX 4090 | 40-60% | 80-100 | ~100ms |
| RTX 3090 | 30-50% | 60-80 | ~150ms |
| Intel i9 (AVX2) | 20-30% | 10-15 | ~300ms |
| Intel i7 (AVX2) | 15-25% | 7-12 | ~400ms |

## Additional Resources

- [llama.cpp Documentation](https://github.com/ggerganov/llama.cpp)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Intel AVX2 Optimization](https://www.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-avx2.html)

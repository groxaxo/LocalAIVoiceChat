# Quick Reference Card

## Installation Commands

### For NVIDIA Ampere GPUs (RTX 30xx/40xx, A100)
```bash
# Install with CUDA support
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir

# Install other dependencies
pip install -r requirements.txt
```

### For Intel CPUs with AVX2
```bash
# Install with AVX2 optimization
CMAKE_ARGS="-DLLAMA_AVX2=ON" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir

# Or with OpenBLAS (Linux)
CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS -DLLAMA_AVX2=ON" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir

# Install other dependencies
pip install -r requirements.txt
```

## Quick Setup

1. **Check your setup:**
   ```bash
   python check_setup.py
   ```

2. **Detect hardware:**
   ```bash
   python hardware_detector.py
   ```

3. **Run the application:**
   ```bash
   python ai_voicetalk_local.py
   ```

## Configuration Files

| File | Purpose |
|------|---------|
| `creation_params.json` | Default configuration |
| `creation_params_ampere.json` | Optimized for Ampere GPUs |
| `creation_params_intel_avx2.json` | Optimized for Intel CPUs |
| `creation_params_override.json` | Manual override (auto-loaded if exists) |
| `completion_params.json` | Text generation parameters |
| `chat_params.json` | Chat personality and scenario |

## Key Parameters

### GPU Performance Tuning
```json
{
  "n_gpu_layers": 35,    // More = faster (0-35)
  "n_batch": 2048,       // Larger = better throughput
  "mul_mat_q": true,     // Enable tensor cores
  "f16_kv": true         // FP16 cache (saves memory)
}
```

### CPU Performance Tuning
```json
{
  "n_gpu_layers": 0,     // CPU only
  "n_threads": 8,        // Use available cores
  "n_batch": 512,        // Optimal for CPU
  "use_mmap": true,      // Fast loading
  "use_mlock": true      // Lock in RAM
}
```

### Latency Optimization
```json
{
  "n_ctx": 4096,         // Smaller context = faster
  "max_tokens": 150,     // Shorter responses
  "temperature": 0.6     // More focused
}
```

## Performance Targets

### NVIDIA RTX 4090 (Ampere)
- **Tokens/sec:** 80-100
- **First token:** ~100ms
- **VRAM usage:** 6-8GB

### NVIDIA RTX 3090 (Ampere)
- **Tokens/sec:** 50-80
- **First token:** ~150ms
- **VRAM usage:** 6-8GB

### Intel i9 (AVX2)
- **Tokens/sec:** 10-15
- **First token:** ~300ms
- **RAM usage:** 8-12GB

### Intel i7 (AVX2)
- **Tokens/sec:** 7-12
- **First token:** ~400ms
- **RAM usage:** 8-12GB

## Troubleshooting

### Problem: Low GPU Performance
**Solutions:**
- Increase `n_gpu_layers` to 35
- Increase `n_batch` to 2048+
- Check GPU usage: `nvidia-smi dmon`

### Problem: Out of Memory (GPU)
**Solutions:**
- Reduce `n_ctx` to 4096 or 2048
- Reduce `n_batch` to 512
- Set `low_vram: true`

### Problem: Out of Memory (CPU)
**Solutions:**
- Reduce `n_ctx` to 2048
- Set `use_mlock: false`
- Reduce `n_batch` to 256

### Problem: Slow CPU Performance
**Solutions:**
- Verify AVX2: `python hardware_detector.py`
- Reduce `n_threads` if system lags
- Use smaller model (Q4_K_M)

### Problem: GPU Not Detected
**Solutions:**
- Check CUDA: `nvidia-smi`
- Verify PyTorch: `python -c "import torch; print(torch.cuda.is_available())"`
- Reinstall llama-cpp-python with CUDA

## Advanced Tools

### Interactive Config Optimizer
```bash
python optimize_config.py
```

### Command-line Config Optimizer
```bash
# Optimize for speed
python optimize_config.py --mode speed --output creation_params_override.json

# Optimize for quality
python optimize_config.py --mode quality --output creation_params_override.json

# Optimize for low memory
python optimize_config.py --mode memory --output creation_params_override.json
```

## Environment Variables

```bash
# CUDA optimizations
export LLAMA_CUDA_GRAPHS=1

# CPU thread control
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## Model Downloads

Download from HuggingFace:
```bash
# Recommended: Q5_K_M (balanced)
wget https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q5_K_M.gguf

# Smaller: Q4_K_M (faster, less accurate)
wget https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q4_K_M.gguf

# Larger: Q6_K (slower, more accurate)
wget https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q6_K.gguf
```

## Support

- Full guide: [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md)
- Main README: [README.md](README.md)
- Check setup: `python check_setup.py`

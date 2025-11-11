"""Hardware Detection Utility for LocalAIVoiceChat
Detects GPU (NVIDIA Ampere) and CPU (Intel AVX2) capabilities to optimize performance.
"""
import platform
import subprocess
import json
import os


def detect_cpu_features():
    """Detect CPU features including AVX2 support."""
    cpu_info = {
        'brand': platform.processor(),
        'cores': os.cpu_count(),
        'avx2': False,
        'avx512': False
    }
    
    system = platform.system()
    
    try:
        if system == 'Linux':
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                if 'avx2' in cpuinfo:
                    cpu_info['avx2'] = True
                if 'avx512' in cpuinfo:
                    cpu_info['avx512'] = True
        elif system == 'Windows':
            # On Windows, check using wmic or other methods
            # This is a simplified check - in practice, we'd use cpuinfo or similar
            try:
                import cpuinfo
                info = cpuinfo.get_cpu_info()
                cpu_info['brand'] = info.get('brand_raw', platform.processor())
                flags = info.get('flags', [])
                cpu_info['avx2'] = 'avx2' in flags
                cpu_info['avx512'] = any('avx512' in flag for flag in flags)
            except ImportError:
                # Fallback - assume modern Intel CPU has AVX2
                if 'Intel' in platform.processor():
                    cpu_info['avx2'] = True
    except Exception as e:
        print(f"Warning: Could not fully detect CPU features: {e}")
    
    return cpu_info


def detect_gpu_info():
    """Detect GPU information, specifically checking for NVIDIA Ampere architecture."""
    gpu_info = {
        'available': False,
        'is_ampere': False,
        'name': None,
        'cuda_version': None,
        'compute_capability': None
    }
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_info['available'] = True
            gpu_info['name'] = torch.cuda.get_device_name(0)
            gpu_info['cuda_version'] = torch.version.cuda
            
            # Get compute capability
            capability = torch.cuda.get_device_capability(0)
            gpu_info['compute_capability'] = f"{capability[0]}.{capability[1]}"
            
            # Ampere GPUs have compute capability 8.0 or 8.6
            # RTX 30xx series: 8.6, A100: 8.0, RTX 40xx (Ada Lovelace): 8.9
            if capability[0] >= 8:
                gpu_info['is_ampere'] = True
                
    except ImportError:
        print("PyTorch not installed, cannot detect GPU")
    except Exception as e:
        print(f"Error detecting GPU: {e}")
    
    return gpu_info


def get_optimal_config():
    """Determine optimal configuration based on detected hardware."""
    cpu_info = detect_cpu_features()
    gpu_info = detect_gpu_info()
    
    config = {
        'cpu': cpu_info,
        'gpu': gpu_info,
        'recommended_config': 'creation_params.json',  # default
        'recommended_threads': max(1, cpu_info['cores'] // 2) if cpu_info['cores'] else 4,
        'recommended_gpu_layers': 0
    }
    
    # Determine best configuration
    if gpu_info['available']:
        if gpu_info['is_ampere']:
            config['recommended_config'] = 'creation_params_ampere.json'
            config['recommended_gpu_layers'] = 35  # Most of model on GPU for Ampere
            config['recommended_threads'] = 4  # Fewer threads when GPU is primary
        else:
            config['recommended_config'] = 'creation_params.json'
            config['recommended_gpu_layers'] = 20  # Conservative for older GPUs
    else:
        # CPU only mode
        if cpu_info['avx2']:
            config['recommended_config'] = 'creation_params_intel_avx2.json'
            config['recommended_threads'] = max(4, cpu_info['cores'] - 2) if cpu_info['cores'] else 6
        config['recommended_gpu_layers'] = 0
    
    return config


def print_hardware_info():
    """Print detected hardware information and recommendations."""
    config = get_optimal_config()
    
    print("=" * 60)
    print("Hardware Detection Results")
    print("=" * 60)
    
    print("\nCPU Information:")
    print(f"  Brand: {config['cpu']['brand']}")
    print(f"  Cores: {config['cpu']['cores']}")
    print(f"  AVX2 Support: {'Yes' if config['cpu']['avx2'] else 'No'}")
    print(f"  AVX512 Support: {'Yes' if config['cpu']['avx512'] else 'No'}")
    
    print("\nGPU Information:")
    if config['gpu']['available']:
        print(f"  Name: {config['gpu']['name']}")
        print(f"  CUDA Version: {config['gpu']['cuda_version']}")
        print(f"  Compute Capability: {config['gpu']['compute_capability']}")
        print(f"  Ampere or newer: {'Yes' if config['gpu']['is_ampere'] else 'No'}")
    else:
        print("  No CUDA GPU detected")
    
    print("\nRecommended Configuration:")
    print(f"  Config File: {config['recommended_config']}")
    print(f"  GPU Layers: {config['recommended_gpu_layers']}")
    print(f"  CPU Threads: {config['recommended_threads']}")
    
    print("=" * 60)
    
    return config


def load_optimal_config(config_override=None):
    """Load the optimal configuration file based on hardware detection."""
    if config_override and os.path.exists(config_override):
        config_file = config_override
    else:
        detection = get_optimal_config()
        config_file = detection['recommended_config']
    
    # Fallback to default if recommended doesn't exist
    if not os.path.exists(config_file):
        config_file = 'creation_params.json'
    
    print(f"Loading configuration from: {config_file}")
    
    with open(config_file, 'r') as f:
        return json.load(f)


if __name__ == '__main__':
    # Run detection and print results
    config = print_hardware_info()
    
    # Save detection results
    with open('hardware_detection.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\nHardware detection saved to 'hardware_detection.json'")

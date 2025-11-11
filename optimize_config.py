#!/usr/bin/env python3
"""
Advanced Configuration Optimizer
Helps users fine-tune their configuration based on specific needs
"""

import json
import os
import argparse
from hardware_detector import get_optimal_config


def load_config(config_file='creation_params.json'):
    """Load a configuration file."""
    with open(config_file, 'r') as f:
        return json.load(f)


def save_config(config, output_file='creation_params_optimized.json'):
    """Save configuration to a file."""
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to: {output_file}")


def optimize_for_speed(config, hw_info):
    """Optimize configuration for maximum speed/minimum latency."""
    print("\nOptimizing for SPEED (lower latency)...")
    
    if hw_info['gpu']['available']:
        # GPU optimizations
        config['n_batch'] = 2048 if hw_info['gpu']['is_ampere'] else 1024
        config['n_gpu_layers'] = 35 if hw_info['gpu']['is_ampere'] else 25
        config['n_threads'] = 4
        config['mul_mat_q'] = True
        config['f16_kv'] = True
    else:
        # CPU optimizations
        config['n_batch'] = 256  # Smaller batch for faster response
        config['n_threads'] = max(2, hw_info['cpu']['cores'] - 2)
        config['use_mmap'] = True
        config['use_mlock'] = True
    
    # Reduce context for faster processing
    config['n_ctx'] = 4096
    
    print("  - Reduced context size to 4096")
    print(f"  - Batch size: {config['n_batch']}")
    print(f"  - GPU layers: {config.get('n_gpu_layers', 0)}")
    
    return config


def optimize_for_quality(config, hw_info):
    """Optimize configuration for maximum quality."""
    print("\nOptimizing for QUALITY (better responses)...")
    
    if hw_info['gpu']['available']:
        config['n_batch'] = 512  # Balanced
        config['n_gpu_layers'] = 35 if hw_info['gpu']['is_ampere'] else 20
        config['n_threads'] = 6
    else:
        config['n_batch'] = 512
        config['n_threads'] = hw_info['cpu']['cores']
    
    # Larger context for better coherence
    config['n_ctx'] = 8192
    
    print("  - Maximum context size: 8192")
    print(f"  - Batch size: {config['n_batch']}")
    
    return config


def optimize_for_memory(config, hw_info):
    """Optimize configuration for low memory usage."""
    print("\nOptimizing for LOW MEMORY usage...")
    
    if hw_info['gpu']['available']:
        # Reduce GPU layers to save VRAM
        config['n_gpu_layers'] = 15
        config['n_batch'] = 256
        config['low_vram'] = True
    else:
        config['n_batch'] = 128
        config['n_threads'] = max(2, hw_info['cpu']['cores'] // 2)
        config['use_mlock'] = False  # Don't lock memory
    
    # Smaller context
    config['n_ctx'] = 2048
    
    print("  - Reduced context to 2048")
    print("  - Lower batch size for memory efficiency")
    print(f"  - GPU layers: {config.get('n_gpu_layers', 0)}")
    
    return config


def optimize_balanced(config, hw_info):
    """Create a balanced configuration."""
    print("\nOptimizing for BALANCED performance...")
    
    if hw_info['gpu']['available']:
        config['n_batch'] = 1024
        config['n_gpu_layers'] = 30 if hw_info['gpu']['is_ampere'] else 20
        config['n_threads'] = 4
        config['mul_mat_q'] = True
    else:
        config['n_batch'] = 512
        config['n_threads'] = hw_info['recommended_threads']
        config['use_mmap'] = True
    
    config['n_ctx'] = 8192
    
    print(f"  - Context: {config['n_ctx']}")
    print(f"  - Batch: {config['n_batch']}")
    print(f"  - GPU layers: {config.get('n_gpu_layers', 0)}")
    
    return config


def interactive_optimizer():
    """Interactive configuration optimizer."""
    print("="*60)
    print("LocalAIVoiceChat Configuration Optimizer")
    print("="*60)
    
    # Get hardware info
    hw_info = get_optimal_config()
    
    print("\nDetected Hardware:")
    print(f"  CPU: {hw_info['cpu']['cores']} cores, AVX2: {hw_info['cpu']['avx2']}")
    if hw_info['gpu']['available']:
        print(f"  GPU: {hw_info['gpu']['name']}")
        print(f"       Ampere: {hw_info['gpu']['is_ampere']}")
    else:
        print("  GPU: Not available (CPU mode)")
    
    # Load base configuration
    base_config = hw_info['recommended_config']
    if not os.path.exists(base_config):
        base_config = 'creation_params.json'
    
    print(f"\nLoading base config: {base_config}")
    config = load_config(base_config)
    
    print("\nSelect optimization goal:")
    print("  1. Speed (lowest latency, faster responses)")
    print("  2. Quality (best responses, higher resource usage)")
    print("  3. Memory (lowest memory usage)")
    print("  4. Balanced (recommended for most users)")
    
    choice = input("\nEnter choice (1-4) [4]: ").strip() or '4'
    
    if choice == '1':
        config = optimize_for_speed(config, hw_info)
    elif choice == '2':
        config = optimize_for_quality(config, hw_info)
    elif choice == '3':
        config = optimize_for_memory(config, hw_info)
    else:
        config = optimize_balanced(config, hw_info)
    
    # Ask for output filename
    output = input("\nOutput filename [creation_params_optimized.json]: ").strip()
    output = output or 'creation_params_optimized.json'
    
    save_config(config, output)
    
    print("\n" + "="*60)
    print("Optimization Complete!")
    print("="*60)
    print(f"\nTo use this configuration:")
    print(f"  1. Copy {output} to creation_params_override.json")
    print(f"  2. Or rename it to creation_params.json")
    print(f"  3. Run: python ai_voicetalk_local.py")


def main():
    parser = argparse.ArgumentParser(
        description='Optimize LocalAIVoiceChat configuration'
    )
    parser.add_argument(
        '--mode',
        choices=['speed', 'quality', 'memory', 'balanced'],
        help='Optimization mode'
    )
    parser.add_argument(
        '--input',
        default='creation_params.json',
        help='Input configuration file'
    )
    parser.add_argument(
        '--output',
        default='creation_params_optimized.json',
        help='Output configuration file'
    )
    
    args = parser.parse_args()
    
    if args.mode:
        # Non-interactive mode
        hw_info = get_optimal_config()
        config = load_config(args.input)
        
        if args.mode == 'speed':
            config = optimize_for_speed(config, hw_info)
        elif args.mode == 'quality':
            config = optimize_for_quality(config, hw_info)
        elif args.mode == 'memory':
            config = optimize_for_memory(config, hw_info)
        else:
            config = optimize_balanced(config, hw_info)
        
        save_config(config, args.output)
    else:
        # Interactive mode
        interactive_optimizer()


if __name__ == '__main__':
    main()

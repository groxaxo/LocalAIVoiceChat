#!/usr/bin/env python3
"""
Performance Benchmark Tool for LocalAIVoiceChat
Tests inference speed and provides optimization recommendations
"""

import time
import json
import sys
from hardware_detector import get_optimal_config


def benchmark_model(model, test_prompts, max_tokens=100):
    """Run benchmark tests on the model."""
    results = []
    
    print("\nRunning benchmarks...")
    print("="*60)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}/{len(test_prompts)}: {prompt[:50]}...")
        
        try:
            # Time the generation
            start_time = time.time()
            tokens_generated = 0
            first_token_time = None
            
            for chunk in model.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                stream=True,
                stop=["</s>"]
            ):
                if first_token_time is None:
                    first_token_time = time.time() - start_time
                
                text = chunk['choices'][0]['text']
                if text:
                    tokens_generated += 1
            
            total_time = time.time() - start_time
            tokens_per_second = tokens_generated / total_time if total_time > 0 else 0
            
            result = {
                'prompt_length': len(prompt),
                'tokens_generated': tokens_generated,
                'total_time': total_time,
                'first_token_time': first_token_time,
                'tokens_per_second': tokens_per_second
            }
            
            results.append(result)
            
            print(f"  First token: {first_token_time:.3f}s")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Tokens: {tokens_generated}")
            print(f"  Speed: {tokens_per_second:.2f} tokens/sec")
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append(None)
    
    return results


def analyze_results(results, hw_config):
    """Analyze benchmark results and provide recommendations."""
    print("\n" + "="*60)
    print("Benchmark Analysis")
    print("="*60)
    
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        print("\nNo valid results to analyze.")
        return
    
    # Calculate averages
    avg_first_token = sum(r['first_token_time'] for r in valid_results) / len(valid_results)
    avg_tokens_per_sec = sum(r['tokens_per_second'] for r in valid_results) / len(valid_results)
    avg_total_time = sum(r['total_time'] for r in valid_results) / len(valid_results)
    
    print(f"\nAverage Performance:")
    print(f"  First Token Latency: {avg_first_token:.3f}s")
    print(f"  Generation Speed: {avg_tokens_per_sec:.2f} tokens/sec")
    print(f"  Average Response Time: {avg_total_time:.2f}s")
    
    # Provide recommendations
    print(f"\nHardware Configuration:")
    if hw_config['gpu']['available']:
        print(f"  GPU: {hw_config['gpu']['name']}")
        print(f"  Ampere: {hw_config['gpu']['is_ampere']}")
    else:
        print(f"  CPU: {hw_config['cpu']['cores']} cores")
        print(f"  AVX2: {hw_config['cpu']['avx2']}")
    
    print(f"\nPerformance Assessment:")
    
    # GPU benchmarks
    if hw_config['gpu']['available']:
        if hw_config['gpu']['is_ampere']:
            target_tps = 60  # RTX 30xx/40xx target
            target_first_token = 0.2
        else:
            target_tps = 30  # Older GPUs
            target_first_token = 0.3
        
        if avg_tokens_per_sec >= target_tps:
            print(f"  ✓ Excellent GPU performance ({avg_tokens_per_sec:.1f} tokens/sec)")
        elif avg_tokens_per_sec >= target_tps * 0.7:
            print(f"  ✓ Good GPU performance ({avg_tokens_per_sec:.1f} tokens/sec)")
        else:
            print(f"  ⚠ Below target GPU performance ({avg_tokens_per_sec:.1f} tokens/sec)")
            print(f"    Target: {target_tps}+ tokens/sec")
    else:
        # CPU benchmarks
        target_tps = 10
        target_first_token = 0.4
        
        if avg_tokens_per_sec >= target_tps:
            print(f"  ✓ Good CPU performance ({avg_tokens_per_sec:.1f} tokens/sec)")
        elif avg_tokens_per_sec >= target_tps * 0.6:
            print(f"  ✓ Acceptable CPU performance ({avg_tokens_per_sec:.1f} tokens/sec)")
        else:
            print(f"  ⚠ Slow CPU performance ({avg_tokens_per_sec:.1f} tokens/sec)")
    
    print(f"\nRecommendations:")
    
    if hw_config['gpu']['available']:
        if avg_tokens_per_sec < 40:
            print("  • Increase n_gpu_layers (try 35 for full offload)")
            print("  • Increase n_batch size (try 2048 for Ampere)")
            print("  • Check GPU utilization with nvidia-smi")
        if avg_first_token > 0.3:
            print("  • Consider reducing n_ctx for faster first token")
            print("  • Ensure GPU is not thermal throttling")
    else:
        if avg_tokens_per_sec < 8:
            print("  • Verify AVX2 support with hardware_detector.py")
            print("  • Adjust n_threads (try 4-8 threads)")
            print("  • Consider using a smaller model (Q4_K_M)")
            print("  • Enable use_mmap and use_mlock")
        if avg_first_token > 0.5:
            print("  • Reduce n_ctx to 4096 or 2048")
            print("  • Reduce n_batch to 256")
    
    # Save results
    results_data = {
        'hardware': hw_config,
        'results': valid_results,
        'summary': {
            'avg_first_token_latency': avg_first_token,
            'avg_tokens_per_second': avg_tokens_per_sec,
            'avg_total_time': avg_total_time
        }
    }
    
    with open('benchmark_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nResults saved to: benchmark_results.json")


def main():
    """Main benchmark function."""
    print("="*60)
    print("LocalAIVoiceChat Performance Benchmark")
    print("="*60)
    
    # Check if model is available
    try:
        with open('creation_params.json', 'r') as f:
            config = json.load(f)
            model_path = config.get('model_path', '')
            
        if not model_path:
            print("\nError: Model path not configured in creation_params.json")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nError loading configuration: {e}")
        sys.exit(1)
    
    # Get hardware info
    print("\nDetecting hardware...")
    hw_config = get_optimal_config()
    
    print(f"CPU: {hw_config['cpu']['cores']} cores, AVX2: {hw_config['cpu']['avx2']}")
    if hw_config['gpu']['available']:
        print(f"GPU: {hw_config['gpu']['name']}")
    else:
        print("GPU: Not available")
    
    # Ask user if they want to proceed
    proceed = input("\nThis will initialize the model and run benchmarks. Continue? (y/n): ")
    if proceed.lower() != 'y':
        print("Benchmark cancelled.")
        return
    
    # Initialize model
    print("\nInitializing model...")
    try:
        import llama_cpp
        
        # Try to import CUDA version if available
        if hw_config['gpu']['available']:
            try:
                import llama_cpp_cuda
                Llama = llama_cpp_cuda.Llama
                print("Using llama_cpp_cuda")
            except:
                Llama = llama_cpp.Llama
                print("Using llama_cpp")
        else:
            Llama = llama_cpp.Llama
            print("Using llama_cpp")
        
        # Load recommended config
        from hardware_detector import load_optimal_config
        creation_params = load_optimal_config()
        
        print(f"Loading model with:")
        print(f"  GPU layers: {creation_params.get('n_gpu_layers', 0)}")
        print(f"  Threads: {creation_params.get('n_threads', 4)}")
        print(f"  Batch size: {creation_params.get('n_batch', 512)}")
        
        start_load = time.time()
        model = Llama(**creation_params)
        load_time = time.time() - start_load
        
        print(f"\nModel loaded in {load_time:.2f} seconds")
        
    except Exception as e:
        print(f"\nError initializing model: {e}")
        print("Make sure the model file exists and dependencies are installed.")
        sys.exit(1)
    
    # Test prompts
    test_prompts = [
        "Hello, how are you today?",
        "What is the capital of France?",
        "Tell me a short joke.",
        "Explain quantum computing in simple terms.",
    ]
    
    # Run benchmarks
    results = benchmark_model(model, test_prompts, max_tokens=50)
    
    # Analyze results
    analyze_results(results, hw_config)
    
    print("\n" + "="*60)
    print("Benchmark Complete!")
    print("="*60)


if __name__ == '__main__':
    main()

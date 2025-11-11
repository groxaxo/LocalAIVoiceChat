#!/usr/bin/env python3
"""
Quick setup checker for LocalAIVoiceChat
Verifies hardware capabilities and configuration
"""

import sys
import os

def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")
    missing = []
    
    try:
        import torch
        print("✓ PyTorch installed")
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available (version {torch.version.cuda})")
            print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("  ℹ CUDA not available (CPU mode will be used)")
    except ImportError:
        print("✗ PyTorch not installed")
        missing.append("torch")
    
    try:
        import llama_cpp
        print("✓ llama-cpp-python installed")
    except ImportError:
        print("✗ llama-cpp-python not installed")
        missing.append("llama-cpp-python")
    
    try:
        from RealtimeSTT import AudioToTextRecorder
        print("✓ RealtimeSTT installed")
    except ImportError:
        print("✗ RealtimeSTT not installed")
        missing.append("RealtimeSTT")
    
    try:
        from RealtimeTTS import TextToAudioStream, CoquiEngine
        print("✓ RealtimeTTS installed")
    except ImportError:
        print("✗ RealtimeTTS not installed")
        missing.append("RealtimeTTS")
    
    return missing

def check_configuration():
    """Check if configuration files exist."""
    print("\nChecking configuration files...")
    
    configs = [
        'creation_params.json',
        'completion_params.json',
        'chat_params.json'
    ]
    
    all_exist = True
    for config in configs:
        if os.path.exists(config):
            print(f"✓ {config} found")
        else:
            print(f"✗ {config} missing")
            all_exist = False
    
    return all_exist

def check_model():
    """Check if the model file path is configured."""
    print("\nChecking model configuration...")
    
    try:
        import json
        with open('creation_params.json', 'r') as f:
            config = json.load(f)
            model_path = config.get('model_path', '')
            
            if model_path:
                print(f"Model path configured: {model_path}")
                if os.path.exists(model_path):
                    print("✓ Model file exists")
                    size_mb = os.path.getsize(model_path) / (1024 * 1024)
                    print(f"  Size: {size_mb:.1f} MB")
                    return True
                else:
                    print("✗ Model file not found at specified path")
                    print("  Please download the model and update the path in creation_params.json")
                    return False
            else:
                print("✗ Model path not configured")
                return False
    except Exception as e:
        print(f"✗ Error checking model: {e}")
        return False

def run_hardware_detection():
    """Run hardware detection if available."""
    print("\nRunning hardware detection...")
    try:
        from hardware_detector import print_hardware_info
        print_hardware_info()
        return True
    except ImportError as e:
        print(f"✗ Could not import hardware_detector: {e}")
        return False
    except Exception as e:
        print(f"✗ Error during hardware detection: {e}")
        return False

def main():
    print("="*60)
    print("LocalAIVoiceChat Setup Checker")
    print("="*60)
    print()
    
    # Check dependencies
    missing_deps = check_dependencies()
    
    # Check configuration
    config_ok = check_configuration()
    
    # Check model
    model_ok = check_model()
    
    # Run hardware detection
    hw_ok = run_hardware_detection()
    
    # Summary
    print("\n" + "="*60)
    print("Setup Summary")
    print("="*60)
    
    if missing_deps:
        print(f"\n⚠ Missing dependencies: {', '.join(missing_deps)}")
        print("\nTo install missing dependencies, run:")
        if 'torch' in missing_deps:
            print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        if 'llama-cpp-python' in missing_deps:
            print("  CMAKE_ARGS=\"-DLLAMA_CUBLAS=on\" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir")
        if 'RealtimeSTT' in missing_deps:
            print("  pip install RealtimeSTT==0.1.7")
        if 'RealtimeTTS' in missing_deps:
            print("  pip install RealtimeTTS==0.2.7")
    
    if not config_ok:
        print("\n⚠ Configuration files are missing")
        print("  Make sure you're in the correct directory")
    
    if not model_ok:
        print("\n⚠ Model not found or not configured")
        print("  Download the model from: https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF")
        print("  Update the model_path in creation_params.json")
    
    if not missing_deps and config_ok and model_ok:
        print("\n✓ All checks passed! You're ready to run:")
        print("  python ai_voicetalk_local.py")
    else:
        print("\n✗ Setup incomplete. Please address the issues above.")
        sys.exit(1)

if __name__ == '__main__':
    main()

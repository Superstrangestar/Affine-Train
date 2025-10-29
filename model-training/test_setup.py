#!/usr/bin/env python3
"""Test script to verify the training environment is set up correctly"""

import sys
from pathlib import Path

print("=" * 80)
print("AFFINE MODEL TRAINING - SETUP VERIFICATION")
print("=" * 80)

# Test Python version
print("\n1. Python Version")
print(f"   Version: {sys.version}")
if sys.version_info >= (3, 10):
    print("   ✓ Python 3.10+ detected")
else:
    print("   ✗ Python 3.10+ required")
    sys.exit(1)

# Test PyTorch
print("\n2. PyTorch")
try:
    import torch
    print(f"   Version: {torch.__version__}")
    print(f"   ✓ PyTorch installed")

    if torch.cuda.is_available():
        print(f"   ✓ CUDA available")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   GPU Memory: {memory_gb:.1f} GB")

        if memory_gb >= 24:
            print("   ✓ Sufficient GPU memory (24GB+)")
        else:
            print("   ⚠ Warning: Less than 24GB VRAM detected")
    else:
        print("   ✗ CUDA not available")
        print("   ⚠ Training will be very slow on CPU")
except ImportError:
    print("   ✗ PyTorch not installed")
    sys.exit(1)

# Test Transformers
print("\n3. Transformers")
try:
    import transformers
    print(f"   Version: {transformers.__version__}")
    print("   ✓ Transformers installed")
except ImportError:
    print("   ✗ Transformers not installed")
    sys.exit(1)

# Test PEFT
print("\n4. PEFT (LoRA)")
try:
    from peft import LoraConfig
    import peft
    print(f"   Version: {peft.__version__}")
    print("   ✓ PEFT installed")
except ImportError:
    print("   ✗ PEFT not installed")
    sys.exit(1)

# Test other dependencies
print("\n5. Other Dependencies")
dependencies = [
    ("accelerate", "accelerate"),
    ("datasets", "datasets"),
    ("wandb", "wandb"),
    ("yaml", "pyyaml"),
    ("scipy", "scipy"),
    ("numpy", "numpy"),
]

missing = []
for module_name, package_name in dependencies:
    try:
        __import__(module_name)
        print(f"   ✓ {package_name}")
    except ImportError:
        print(f"   ✗ {package_name} not installed")
        missing.append(package_name)

if missing:
    print(f"\n   Missing packages: {', '.join(missing)}")
    print(f"   Install with: pip install {' '.join(missing)}")

# Check base model
print("\n6. Base Model")
base_model_path = Path("../Affine-0004")
if base_model_path.exists():
    print(f"   Path: {base_model_path}")

    required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    all_present = all((base_model_path / f).exists() for f in required_files)

    if all_present:
        print("   ✓ Affine-0004 model found")
    else:
        print("   ⚠ Some model files missing")
else:
    print(f"   ✗ Base model not found at {base_model_path}")
    print("   Please ensure Affine-0004 is in the parent directory")

# Check directories
print("\n7. Project Structure")
required_dirs = ["src", "scripts", "data_cache", "checkpoints", "models", "logs"]
for dir_name in required_dirs:
    dir_path = Path(dir_name)
    if dir_path.exists():
        print(f"   ✓ {dir_name}/")
    else:
        print(f"   Creating {dir_name}/")
        dir_path.mkdir(parents=True, exist_ok=True)

# Check config file
print("\n8. Configuration")
config_path = Path("config.yaml")
if config_path.exists():
    print(f"   ✓ config.yaml found")

    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"   Model: {config.get('model', {}).get('base_model_path', 'N/A')}")
        print(f"   LoRA: {config.get('training', {}).get('use_lora', 'N/A')}")
        print(f"   Epochs: {config.get('training', {}).get('num_epochs', 'N/A')}")
    except Exception as e:
        print(f"   ⚠ Error loading config: {e}")
else:
    print(f"   ✗ config.yaml not found")

# Check disk space
print("\n9. Disk Space")
import shutil
total, used, free = shutil.disk_usage(".")
free_gb = free / 1e9
print(f"   Free: {free_gb:.1f} GB")
if free_gb >= 100:
    print("   ✓ Sufficient disk space (100GB+)")
else:
    print("   ⚠ Warning: Less than 100GB free")

# Summary
print("\n" + "=" * 80)
print("SETUP VERIFICATION COMPLETE")
print("=" * 80)

if torch.cuda.is_available() and base_model_path.exists():
    print("\n✓ Your environment is ready for training!")
    print("\nNext steps:")
    print("  1. Review config.yaml")
    print("  2. Run: python scripts/run_full_pipeline.py")
    print("  or")
    print("  3. Run individual steps:")
    print("     python scripts/1_collect_data.py")
    print("     python scripts/2_train_sft.py")
    print("     python scripts/3_train_rl.py")
    print("     python scripts/4_evaluate.py")
else:
    print("\n⚠ Some issues detected. Please resolve them before training.")
    if not torch.cuda.is_available():
        print("  - CUDA not available")
    if not base_model_path.exists():
        print("  - Base model not found")

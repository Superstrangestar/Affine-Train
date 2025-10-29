# Quickstart Guide

Get started with Affine model training in 5 minutes!

## Prerequisites

- Python 3.10+
- CUDA-capable GPU with 24GB+ VRAM
- 100GB free disk space

## Installation

```bash
cd model-training
pip install -r requirements.txt
pip install -e .
```

## Run Training (Simplified)

### Option 1: Full Automated Pipeline

```bash
python scripts/run_full_pipeline.py
```

This runs everything automatically:
- ✅ Data collection
- ✅ Supervised fine-tuning
- ✅ RL training
- ✅ Evaluation
- ✅ Deployment

**Estimated time:** 14-26 hours (depending on GPU)

### Option 2: Quick Test Run

For testing the pipeline quickly with minimal data:

```bash
# Edit config.yaml to reduce samples
nano config.yaml

# Change these values:
# data.affine.num_samples: 100  (instead of 10000)
# data.agentgym.num_episodes: 50  (instead of 5000)
# training.num_epochs: 1  (instead of 3)

# Run pipeline
python scripts/run_full_pipeline.py
```

**Estimated time:** 2-4 hours

## Step-by-Step Guide

If you prefer to run each step manually:

### Step 1: Data Collection (~1 hour)

```bash
python scripts/1_collect_data.py
```

This collects training data from all 8 environments and saves to `data_cache/`.

### Step 2: Supervised Fine-Tuning (~4-8 hours)

```bash
python scripts/2_train_sft.py
```

This fine-tunes the model on reasoning tasks (SAT, ABD, DED).

Output: `models/sft_final/`

### Step 3: RL Training (~8-16 hours)

```bash
python scripts/3_train_rl.py
```

This applies PPO to improve performance on interactive agent tasks.

Output: `models/rl_final/`

### Step 4: Evaluation (~1 hour)

```bash
python scripts/4_evaluate.py
```

This evaluates the final model on all 8 environments.

Output: `logs/evaluation_results.json`

### Step 5: Deploy to HuggingFace

```bash
export HF_TOKEN="your_huggingface_token"
python scripts/5_deploy.py
```

This uploads your trained model to HuggingFace Hub.

## Verify Your Setup

Test that everything is installed correctly:

```python
# test_setup.py
import torch
import transformers
from peft import LoraConfig
print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ Transformers: {transformers.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

## Expected Results

After training, you should see:

### Evaluation Metrics
```
Weighted Accuracy: 0.65-0.75
Weighted Mean Score: 0.60-0.70
Overall Success Rate: 0.70-0.80
```

### Per-Environment Scores
- SAT: 0.50-0.70
- ABD: 0.40-0.60
- DED: 0.45-0.65
- WebShop: 0.60-0.80
- AlfWorld: 0.70-0.85
- BabyAI: 0.75-0.90
- SciWorld: 0.55-0.75
- TextCraft: 0.65-0.85

*Note: These are approximate ranges. Actual performance depends on training configuration and randomness.*

## Common Issues

### GPU Out of Memory

```yaml
# In config.yaml, reduce batch size:
training:
  batch_size: 1
  gradient_accumulation_steps: 16
```

### Slow Training

```yaml
# Enable optimizations:
model:
  use_flash_attention: true
training:
  bf16: true
  gradient_checkpointing: true
```

### Missing Base Model

```bash
# Ensure Affine-0004 is in parent directory:
ls ../Affine-0004/
# Should show: config.json, model-*.safetensors, tokenizer files
```

## Next Steps

1. **Monitor Training**: Set up W&B tracking
2. **Tune Hyperparameters**: Experiment with learning rates, batch sizes
3. **Custom Environments**: Add your own evaluation tasks
4. **Deploy**: Share your model on HuggingFace Hub

## Getting Help

- Check `logs/` for detailed error messages
- See `README.md` for full documentation
- Review `config.yaml` comments for parameter explanations

---

**Ready to train! 🚀**

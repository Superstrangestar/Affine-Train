# Affine Model Training

Complete training pipeline for fine-tuning Affine-0004 (Qwen3-4B) model to achieve higher scores on reasoning and interactive agent tasks.

## Overview

This project provides a comprehensive training infrastructure to:
- Fine-tune the Affine-0004 model using supervised learning and reinforcement learning
- Train on 8 diverse environments (SAT, ABD, DED, WebShop, AlfWorld, BabyAI, SciWorld, TextCraft)
- Evaluate performance using the same metrics as Affine validators
- Deploy the trained model to HuggingFace Hub

## Features

✨ **Comprehensive Training Pipeline**
- Supervised Fine-Tuning (SFT) on reasoning tasks
- Reinforcement Learning (PPO) on interactive agent tasks
- Automatic data collection and preprocessing
- Multi-environment training and evaluation

🚀 **Efficient Fine-Tuning**
- LoRA (Low-Rank Adaptation) for parameter-efficient training
- Flash Attention 2 for fast inference
- Gradient checkpointing for memory efficiency
- Mixed precision training (BF16)

📊 **Advanced Evaluation**
- Pareto dominance scoring (aligned with validators)
- Bayesian confidence intervals
- Per-environment and aggregated metrics
- Comparison with baseline models

🤗 **HuggingFace Integration**
- Automatic model card generation
- One-command deployment
- Git LFS support for large files

## Project Structure

```
model-training/
├── config.yaml                 # Main configuration file
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
├── README.md                   # This file
│
├── src/                        # Source code
│   ├── data/                   # Data collection modules
│   │   ├── affine_data.py      # Affine tasks (SAT, ABD, DED)
│   │   ├── agentgym_data.py    # AgentGym tasks (RL environments)
│   │   └── dataset.py          # PyTorch dataset wrappers
│   │
│   ├── training/               # Training modules
│   │   ├── model_loader.py     # Model loading utilities
│   │   ├── sft_trainer.py      # Supervised fine-tuning
│   │   └── rl_trainer.py       # Reinforcement learning (PPO)
│   │
│   ├── evaluation/             # Evaluation modules
│   │   ├── evaluator.py        # Model evaluator
│   │   └── metrics.py          # Metrics computation
│   │
│   ├── deployment/             # Deployment modules
│   │   └── hf_deploy.py        # HuggingFace deployment
│   │
│   └── utils/                  # Utility modules
│       ├── logger.py           # Logging setup
│       └── config.py           # Configuration utilities
│
├── scripts/                    # Training scripts
│   ├── 1_collect_data.py       # Data collection
│   ├── 2_train_sft.py          # Supervised fine-tuning
│   ├── 3_train_rl.py           # RL training
│   ├── 4_evaluate.py           # Model evaluation
│   ├── 5_deploy.py             # HuggingFace deployment
│   └── run_full_pipeline.py    # Complete pipeline
│
├── data_cache/                 # Cached training data
├── checkpoints/                # Training checkpoints
├── models/                     # Saved models
└── logs/                       # Training logs
```

## Quick Start

### 1. Installation

```bash
cd model-training

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### 2. Configuration

Edit `config.yaml` to customize training parameters:

```yaml
model:
  base_model_path: "../Affine-0004"

training:
  num_epochs: 3
  batch_size: 2
  learning_rate: 2.0e-5
  use_lora: true
  lora_r: 64

# ... see config.yaml for full options
```

### 3. Run Training Pipeline

**Option A: Complete Pipeline (Recommended)**

```bash
python scripts/run_full_pipeline.py
```

This runs all steps automatically:
1. Data collection
2. Supervised fine-tuning
3. RL training
4. Evaluation
5. Deployment

**Option B: Step-by-Step**

```bash
# Step 1: Collect training data
python scripts/1_collect_data.py

# Step 2: Supervised fine-tuning
python scripts/2_train_sft.py

# Step 3: RL training (optional but recommended)
python scripts/3_train_rl.py

# Step 4: Evaluate model
python scripts/4_evaluate.py

# Step 5: Deploy to HuggingFace
export HF_TOKEN="your_token_here"
python scripts/5_deploy.py
```

## Training Details

### Supervised Fine-Tuning (SFT)

Trains the model on:
- **SAT Solving**: Boolean satisfiability problems
- **Abduction (ABD)**: Reverse engineering inputs from code
- **Deduction (DED)**: Code generation from problem descriptions

Configuration:
- LoRA rank: 64
- Learning rate: 2e-5
- Batch size: 2 (gradient accumulation: 8)
- Epochs: 3

### Reinforcement Learning (RL)

Trains the model using PPO on interactive environments:
- **WebShop**: E-commerce product search
- **AlfWorld**: Household task completion
- **BabyAI**: Grid-world navigation
- **SciWorld**: Scientific experiments
- **TextCraft**: Minecraft crafting

Configuration:
- Algorithm: PPO (Proximal Policy Optimization)
- Clip range: 0.2
- GAE lambda: 0.95
- Gamma: 0.99

### Evaluation

Evaluates model performance using:
- Binary accuracy (for Affine tasks)
- Mean score and standard deviation
- Bayesian confidence intervals (80%)
- Success rate per environment
- Pareto dominance scoring

## Hardware Requirements

**Minimum:**
- GPU: 24GB VRAM (NVIDIA A10, RTX 3090, RTX 4090)
- RAM: 32GB
- Disk: 100GB free space

**Recommended:**
- GPU: 40GB+ VRAM (NVIDIA A100, H100)
- RAM: 64GB
- Disk: 200GB SSD

**Training Time Estimates:**
- Data collection: ~1 hour
- SFT training: ~4-8 hours (depending on GPU)
- RL training: ~8-16 hours
- Evaluation: ~1 hour
- Total: ~14-26 hours

## Configuration Options

### Key Configuration Parameters

**Model:**
- `base_model_path`: Path to Affine-0004 model
- `use_flash_attention`: Enable Flash Attention 2
- `torch_dtype`: Precision (bfloat16 recommended)

**Training:**
- `num_epochs`: Number of training epochs
- `batch_size`: Per-device batch size
- `learning_rate`: Learning rate
- `use_lora`: Enable LoRA fine-tuning
- `lora_r`: LoRA rank (higher = more parameters)

**Data:**
- `num_samples`: Number of training samples per environment
- `train_split`: Training split ratio (default: 0.9)

**RL Training:**
- `ppo_epochs`: PPO update epochs
- `num_rollouts`: Number of rollouts to collect
- `clip_range`: PPO clip range

**Evaluation:**
- `num_eval_samples`: Samples per environment
- `min_score_threshold`: Minimum acceptable score

## Deployment

### Prerequisites

```bash
# Set HuggingFace token
export HF_TOKEN="your_token_here"

# Or add to .env file
echo "HF_TOKEN=your_token_here" > .env
```

### Deploy to HuggingFace Hub

```bash
python scripts/5_deploy.py --model models/rl_final
```

This will:
1. Create a model card with training details
2. Upload model files to HuggingFace Hub
3. Configure Git LFS for large files
4. Return the model URL

### Configuration

Edit `config.yaml`:

```yaml
huggingface:
  repo_name: "your-username/affine-model"
  private: false
  create_model_card: true
```

## Monitoring

### Weights & Biases (W&B)

Enable W&B tracking in `config.yaml`:

```yaml
tracking:
  use_wandb: true
  wandb_project: "affine-training"
  wandb_entity: "your-username"
```

### Logs

Training logs are saved to `logs/`:
- `data-collection_*.log`
- `sft-training_*.log`
- `rl-training_*.log`
- `evaluation_*.log`
- `deployment_*.log`

## Troubleshooting

### Out of Memory (OOM)

**Solutions:**
1. Reduce batch size in `config.yaml`
2. Enable gradient checkpointing
3. Use 8-bit or 4-bit quantization
4. Reduce LoRA rank

```yaml
training:
  batch_size: 1
  gradient_checkpointing: true
```

### Slow Training

**Solutions:**
1. Enable Flash Attention 2
2. Use BF16 mixed precision
3. Increase batch size (if memory allows)
4. Reduce gradient accumulation steps

### Data Collection Errors

**Solutions:**
1. Check internet connection (for R2 dataset)
2. Verify test-code module is available
3. Use cached data if available

## Advanced Usage

### Custom Environments

Add custom environments in `config.yaml`:

```yaml
data:
  affine:
    environments:
      - "affine:sat"
      - "affine:abd"
      - "affine:ded"
      - "custom:my_env"  # Add your environment
```

### Hyperparameter Tuning

Create multiple config files:

```bash
# configs/experiment1.yaml
# configs/experiment2.yaml

python scripts/run_full_pipeline.py --config configs/experiment1.yaml
```

### Resume Training

```yaml
training:
  resume_from_checkpoint: "checkpoints/checkpoint-1000"
```

## Performance Optimization Tips

1. **Use LoRA**: Reduces trainable parameters by 99%
2. **Flash Attention 2**: 2-3x faster training
3. **Gradient Checkpointing**: Reduces memory by ~40%
4. **BF16 Training**: Faster than FP32, more stable than FP16
5. **Larger Batch Size**: Better GPU utilization
6. **Multi-GPU**: Use `accelerate` for distributed training

## Citation

```bibtex
@misc{affine-training-2024,
  title={Affine Model Training Pipeline},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/affine-training}
}
```

## License

Apache 2.0

## Acknowledgments

- **Base Model**: [Affine-0004](https://huggingface.co/kiwikiw/Affine-0004) (Qwen3-4B)
- **Training Framework**: HuggingFace Transformers, PEFT, TRL
- **Evaluation**: Affine Subnet (Bittensor)
- **Environments**: AgentGym, custom Affine tasks

## Support

For issues and questions:
- GitHub Issues: [your-repo/issues]
- Discord: [your-discord-link]
- Email: your-email@example.com

---

**Happy Training! 🚀**

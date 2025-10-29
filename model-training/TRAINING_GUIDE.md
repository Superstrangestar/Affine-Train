# Comprehensive Training Guide

This guide provides detailed information about training the Affine model to achieve optimal performance.

## Table of Contents

1. [Training Strategy](#training-strategy)
2. [Data Collection](#data-collection)
3. [Supervised Fine-Tuning](#supervised-fine-tuning)
4. [Reinforcement Learning](#reinforcement-learning)
5. [Evaluation](#evaluation)
6. [Hyperparameter Tuning](#hyperparameter-tuning)
7. [Best Practices](#best-practices)

## Training Strategy

The training pipeline uses a two-stage approach:

### Stage 1: Supervised Fine-Tuning (SFT)
- **Purpose**: Learn basic reasoning patterns and task formats
- **Data**: Labeled examples from Affine tasks (SAT, ABD, DED)
- **Method**: Standard supervised learning with cross-entropy loss
- **Duration**: 3 epochs (~4-8 hours)

### Stage 2: Reinforcement Learning (RL)
- **Purpose**: Optimize for interactive decision-making and multi-step reasoning
- **Data**: Episodes from AgentGym environments
- **Method**: PPO (Proximal Policy Optimization)
- **Duration**: 100 rollouts (~8-16 hours)

### Why This Approach?

1. **SFT First**: Provides a strong initialization for RL training
2. **RL Second**: Fine-tunes decision-making and exploration strategies
3. **Combined**: Achieves better performance than either approach alone

## Data Collection

### Affine Tasks

The data collector generates/retrieves training samples for:

#### 1. SAT Solving
- **Task**: Find satisfying assignments for k-SAT formulas
- **Generation**: Random k-SAT instances near phase transition
- **Parameters**:
  - n_vars: 10-20
  - k: 3-5
  - clause ratio: 4.2 (near satisfiability threshold)
- **Format**: Boolean formula → variable assignment

Example:
```
Input: (x1 ∨ x2 ∨ ¬x3) ∧ (¬x1 ∨ x3) ∧ ...
Output: x1=True, x2=False, x3=True, ...
```

#### 2. Abduction (ABD)
- **Task**: Reverse engineer inputs from code and outputs
- **Source**: R2 dataset (satpalsr/rl-python)
- **Format**: Code + expected output → input

Example:
```
Code:
    def solve():
        x = int(input())
        print(x * 2)

Expected Output: 10
Required Input: 5
```

#### 3. Deduction (DED)
- **Task**: Generate code from problem descriptions
- **Source**: R2 dataset (satpalsr/rl-python)
- **Format**: Problem description → complete program

Example:
```
Problem: Read two integers and print their sum
Output:
    a = int(input())
    b = int(input())
    print(a + b)
```

### AgentGym Tasks

The data collector generates synthetic episodes for:

#### 1. WebShop
- **Task**: Navigate e-commerce site to find and purchase items
- **Actions**: search[query], click[item]
- **Rewards**: 0.1 (search) + 0.3 (click) + 1.0 (purchase)

#### 2. AlfWorld
- **Task**: Complete household tasks
- **Actions**: goto, take, put, open, close, heat, cool, clean, examine
- **Rewards**: 0.2 per step + 1.0 on completion

#### 3. BabyAI
- **Task**: Navigate grid world and complete goals
- **Actions**: move forward, turn left/right, pick up, toggle
- **Rewards**: 1.0 on goal completion

#### 4. SciWorld
- **Task**: Perform scientific experiments
- **Actions**: goto, take, use, activate, pour, mix, etc.
- **Rewards**: 0.15 per step + 1.0 on completion

#### 5. TextCraft
- **Task**: Craft items in Minecraft-like environment
- **Actions**: get[resource], craft[item]
- **Rewards**: 0.15 per step + 1.0 on completion

### Data Collection Configuration

```yaml
data:
  affine:
    num_samples: 10000  # Total samples across 3 environments
    dataset: "satpalsr/rl-python"
    buffer_size: 5
    environments:
      - "affine:sat"
      - "affine:abd"
      - "affine:ded"

  agentgym:
    num_episodes: 5000  # Total episodes across 5 environments
    max_episode_length: 50
    environments:
      - "agentgym:webshop"
      - "agentgym:alfworld"
      - "agentgym:babyai"
      - "agentgym:sciworld"
      - "agentgym:textcraft"
```

## Supervised Fine-Tuning

### Training Process

1. **Model Loading**: Load Affine-0004 with LoRA adapters
2. **Data Preparation**: Create train/val split (90/10)
3. **Training**: Standard causal language modeling
4. **Checkpointing**: Save best model based on eval loss

### LoRA Configuration

```yaml
training:
  use_lora: true
  lora_r: 64              # Rank (higher = more parameters)
  lora_alpha: 128         # Scaling factor
  lora_dropout: 0.05      # Dropout rate
  lora_target_modules:    # Which layers to adapt
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
```

**Why LoRA?**
- Reduces trainable parameters from 4B to ~40M (99% reduction)
- Faster training and lower memory usage
- Prevents catastrophic forgetting of base model knowledge
- Easy to merge back into full model for deployment

### Training Hyperparameters

```yaml
training:
  num_epochs: 3
  batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-5
  warmup_steps: 100
  weight_decay: 0.01
  max_grad_norm: 1.0

  # Schedule
  scheduler_type: "cosine"
  min_lr_ratio: 0.1

  # Precision
  bf16: true
  fp16: false

  # Memory optimization
  gradient_checkpointing: true
```

### Monitoring Training

Key metrics to watch:
- **Train Loss**: Should decrease steadily
- **Eval Loss**: Should decrease but may plateau
- **Perplexity**: Lower is better
- **Learning Rate**: Should follow cosine schedule

Example training curve:
```
Epoch 1: train_loss=1.85, eval_loss=1.72
Epoch 2: train_loss=1.45, eval_loss=1.58
Epoch 3: train_loss=1.28, eval_loss=1.52
```

## Reinforcement Learning

### PPO Algorithm

The RL trainer uses Proximal Policy Optimization (PPO):

1. **Collect Rollouts**: Generate episodes using current policy
2. **Compute Advantages**: Use GAE (Generalized Advantage Estimation)
3. **Policy Update**: Optimize clipped surrogate objective
4. **Value Update**: Train value network to predict returns
5. **Repeat**: Iterate until convergence

### PPO Configuration

```yaml
rl_training:
  algorithm: "ppo"
  ppo_epochs: 4              # Update epochs per rollout
  num_rollouts: 100          # Total rollouts
  rollout_batch_size: 4      # Parallel rollouts

  # PPO hyperparameters
  clip_range: 0.2            # Clip ratio for policy
  vf_coef: 0.5               # Value loss coefficient
  entropy_coef: 0.01         # Entropy bonus
  gae_lambda: 0.95           # GAE lambda
  gamma: 0.99                # Discount factor

  # Optimization
  learning_rate: 1.0e-5
  max_grad_norm: 1.0
```

### Reward Shaping

Each environment provides rewards:
- **Step rewards**: Small positive values for progress
- **Completion rewards**: Large positive value for success
- **Failure penalties**: Small negative values for errors

Example (AlfWorld):
```
Step 1: goto[table] → reward=0.2
Step 2: take[mug] → reward=0.2
Step 3: goto[sink] → reward=0.2
Step 4: clean[mug] → reward=0.2
Step 5: goto[coffee machine] → reward=0.2
Step 6: put[mug] → reward=1.0 (task complete!)
Total: 2.0
```

### Value Network

A separate value network estimates future returns:
- **Architecture**: 2-layer MLP
- **Input**: Hidden states from language model
- **Output**: Scalar value estimate
- **Loss**: MSE between predicted and actual returns

## Evaluation

### Metrics

#### Per-Environment Metrics
- **Accuracy**: Percentage of correct predictions (binary tasks)
- **Mean Score**: Average score across all samples
- **Standard Deviation**: Score variance
- **Median Score**: 50th percentile score
- **Confidence Interval**: Bayesian 80% CI using Beta distribution
- **Success Rate**: Percentage of non-zero scores

#### Aggregate Metrics
- **Weighted Accuracy**: Accuracy weighted by samples per environment
- **Weighted Mean Score**: Score weighted by samples per environment
- **Overall Success Rate**: Average success rate across environments

### Evaluation Process

1. **Load Model**: Load trained model (RL → SFT → Base)
2. **Generate Responses**: Run model on test samples
3. **Compute Metrics**: Calculate per-environment and aggregate metrics
4. **Save Results**: Export to JSON for analysis

### Confidence Intervals

Uses Bayesian approach with Jeffrey's prior:

```python
alpha = 0.5 + successes
beta = 0.5 + (trials - successes)
lower = beta.ppf(0.10, alpha, beta)  # 10th percentile
upper = beta.ppf(0.90, alpha, beta)  # 90th percentile
```

This provides 80% confidence intervals aligned with validator scoring.

### Pareto Dominance

The evaluation system checks Pareto dominance:
- Model A dominates B if: A ≥ B on all environments AND A > B on at least one
- Models on the Pareto frontier are not dominated by any other model
- This aligns with the Affine subnet validation criteria

## Hyperparameter Tuning

### Learning Rate

**SFT:** 2e-5 is a good default for LoRA fine-tuning

Try:
- Lower (1e-5): More stable but slower convergence
- Higher (5e-5): Faster but risk of instability

**RL:** 1e-5 is a good default for PPO

Try:
- Lower (5e-6): More stable policy updates
- Higher (2e-5): Faster learning but higher variance

### Batch Size

**Trade-off:** Larger batch = better GPU utilization but more memory

Options:
```yaml
# Small GPU (24GB)
batch_size: 1
gradient_accumulation_steps: 16

# Medium GPU (40GB)
batch_size: 2
gradient_accumulation_steps: 8

# Large GPU (80GB)
batch_size: 4
gradient_accumulation_steps: 4
```

### LoRA Rank

**Trade-off:** Higher rank = more parameters = better performance but slower

Options:
- `lora_r: 32` - Fast, lightweight (20M params)
- `lora_r: 64` - Balanced (40M params) **[Default]**
- `lora_r: 128` - High capacity (80M params)

### PPO Hyperparameters

**Clip Range:** Controls how much the policy can change per update
- Lower (0.1): More conservative updates
- Higher (0.3): More aggressive updates
- Default: 0.2

**GAE Lambda:** Controls bias-variance trade-off in advantage estimation
- Lower (0.9): Lower variance, higher bias
- Higher (0.99): Higher variance, lower bias
- Default: 0.95

## Best Practices

### 1. Start with SFT

Always run SFT before RL:
```bash
python scripts/2_train_sft.py
python scripts/3_train_rl.py  # Uses SFT output
```

### 2. Monitor GPU Memory

```python
import torch
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

### 3. Use Mixed Precision

```yaml
training:
  bf16: true  # Faster and more stable than fp16
```

### 4. Enable Gradient Checkpointing

Reduces memory usage by ~40%:
```yaml
training:
  gradient_checkpointing: true
```

### 5. Save Checkpoints Frequently

```yaml
training:
  save_steps: 1000
  save_total_limit: 3  # Keep only 3 recent checkpoints
```

### 6. Track Experiments

Use W&B for experiment tracking:
```yaml
tracking:
  use_wandb: true
  wandb_project: "affine-training"
```

### 7. Evaluate Regularly

Run evaluation after each training stage:
```bash
python scripts/4_evaluate.py --model models/sft_final
python scripts/4_evaluate.py --model models/rl_final
```

### 8. Compare with Baseline

Keep baseline results for comparison:
```bash
# Evaluate base model
python scripts/4_evaluate.py --model ../Affine-0004
mv logs/evaluation_results.json logs/baseline_results.json

# Evaluate your model
python scripts/4_evaluate.py --model models/rl_final
```

### 9. Test on Individual Environments

Debug issues by testing single environments:
```python
# In config.yaml
evaluation:
  environments:
    - "affine:sat"  # Test only SAT
```

### 10. Use Version Control

Track your experiments:
```bash
git init
git add config.yaml
git commit -m "Initial configuration"
git tag v1.0
```

## Common Issues and Solutions

### Issue: Training Loss Not Decreasing

**Possible Causes:**
- Learning rate too low
- Insufficient training data
- Model not learning task format

**Solutions:**
- Increase learning rate to 5e-5
- Collect more data
- Check data formatting

### Issue: Overfitting (High Train Accuracy, Low Val Accuracy)

**Solutions:**
- Increase dropout
- Reduce LoRA rank
- Add more training data
- Early stopping

### Issue: RL Not Improving

**Solutions:**
- Increase number of rollouts
- Adjust reward shaping
- Reduce clip range for more conservative updates
- Ensure SFT model is good starting point

### Issue: Out of Memory

**Solutions:**
- Reduce batch size
- Enable gradient checkpointing
- Use 8-bit quantization
- Reduce sequence length

## Performance Benchmarks

### Expected Training Times (A100 40GB)

| Stage | Time | Memory |
|-------|------|--------|
| Data Collection | 1 hour | 10 GB |
| SFT Training | 4 hours | 35 GB |
| RL Training | 8 hours | 38 GB |
| Evaluation | 1 hour | 30 GB |
| **Total** | **14 hours** | **38 GB peak** |

### Expected Performance Improvements

| Metric | Base Model | After SFT | After RL |
|--------|-----------|-----------|----------|
| SAT Accuracy | 0.40 | 0.55 | 0.60 |
| ABD Accuracy | 0.30 | 0.45 | 0.50 |
| DED Accuracy | 0.35 | 0.50 | 0.55 |
| AgentGym Score | 0.50 | 0.55 | 0.70 |
| **Overall** | **0.39** | **0.51** | **0.59** |

*Note: Actual results may vary based on random seed and configuration.*

## Advanced Topics

### Multi-GPU Training

```bash
# Using accelerate
accelerate config
accelerate launch scripts/2_train_sft.py
```

### Quantization

```yaml
model:
  load_in_8bit: true  # 8-bit quantization
  # or
  load_in_4bit: true  # 4-bit quantization (more aggressive)
```

### Custom Reward Functions

Edit `src/training/rl_trainer.py` to implement custom reward shaping.

### Curriculum Learning

Train on easier tasks first, gradually increase difficulty.

---

**Happy Training! 🎯**

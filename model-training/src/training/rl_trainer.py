"""Reinforcement Learning trainer for AgentGym tasks using PPO"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import wandb
from tqdm import tqdm
import numpy as np

from .model_loader import load_model_and_tokenizer, save_model, count_parameters
from data.dataset import RLDataset


@dataclass
class RolloutBuffer:
    """Buffer for storing rollout data"""
    observations: List[str]
    actions: List[str]
    log_probs: List[torch.Tensor]
    values: List[torch.Tensor]
    rewards: List[float]
    dones: List[bool]
    advantages: List[torch.Tensor]
    returns: List[torch.Tensor]

    def __init__(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.advantages = []
        self.returns = []

    def clear(self):
        """Clear all buffers"""
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
        self.advantages.clear()
        self.returns.clear()

    def compute_advantages(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        """Compute advantages using GAE"""
        advantages = []
        returns = []
        gae = 0
        next_value = 0

        for i in reversed(range(len(self.rewards))):
            delta = self.rewards[i] + gamma * next_value * (1 - self.dones[i]) - self.values[i].item()
            gae = delta + gamma * gae_lambda * (1 - self.dones[i]) * gae
            advantages.insert(0, torch.tensor(gae))
            returns.insert(0, torch.tensor(gae + self.values[i].item()))
            next_value = self.values[i].item()

        self.advantages = advantages
        self.returns = returns


class ValueNetwork(nn.Module):
    """Value network for PPO"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, hidden_states):
        """Forward pass"""
        return self.value_head(hidden_states).squeeze(-1)


class RLTrainer:
    """Reinforcement learning trainer using PPO"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = config.get("model", {})
        self.rl_config = config.get("rl_training", {})
        self.data_config = config.get("data", {}).get("agentgym", {})
        self.tracking_config = config.get("tracking", {})

        self.model = None
        self.tokenizer = None
        self.value_net = None
        self.dataset = None
        self.rollout_buffer = RolloutBuffer()

        # PPO hyperparameters
        self.gamma = self.rl_config.get("gamma", 0.99)
        self.gae_lambda = self.rl_config.get("gae_lambda", 0.95)
        self.clip_range = self.rl_config.get("clip_range", 0.2)
        self.vf_coef = self.rl_config.get("vf_coef", 0.5)
        self.entropy_coef = self.rl_config.get("entropy_coef", 0.01)
        self.ppo_epochs = self.rl_config.get("ppo_epochs", 4)
        self.num_rollouts = self.rl_config.get("num_rollouts", 100)

    def setup(self):
        """Initialize model, tokenizer, and value network"""
        print("=" * 80)
        print("RL TRAINING SETUP")
        print("=" * 80)

        # Load model and tokenizer
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_path=self.model_config["base_model_path"],
            use_lora=True,
            lora_config={
                "r": self.rl_config.get("lora_r", 64),
                "lora_alpha": self.rl_config.get("lora_alpha", 128),
                "lora_dropout": self.rl_config.get("lora_dropout", 0.05),
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            },
            device_map=self.model_config.get("device_map", "auto"),
            torch_dtype=self.model_config.get("torch_dtype", "bfloat16"),
        )

        # Create value network
        hidden_size = self.model.config.hidden_size
        self.value_net = ValueNetwork(hidden_size)
        self.value_net = self.value_net.to(self.model.device)

        # Print parameter counts
        param_info = count_parameters(self.model)
        print(f"\nModel Parameters:")
        print(f"  Total: {param_info['total_parameters']:,}")
        print(f"  Trainable: {param_info['trainable_parameters']:,}")
        print(f"  Trainable %: {param_info['trainable_percentage']:.2f}%")

        # Load dataset
        print("\nLoading RL dataset...")
        environments = self.data_config.get("environments", [])
        data_dir = Path("data_cache/agentgym")

        self.dataset = RLDataset(
            data_dir=data_dir,
            environments=environments,
            tokenizer=self.tokenizer,
            max_length=4096,
            format_type="react"
        )

        print(f"Dataset size: {len(self.dataset)} transitions")

        # Initialize W&B if configured
        if self.tracking_config.get("use_wandb", False):
            wandb.init(
                project=self.tracking_config.get("wandb_project", "affine-training"),
                entity=self.tracking_config.get("wandb_entity"),
                name=f"{self.tracking_config.get('experiment_name', 'rl-training')}-rl",
                config=self.config
            )

    def get_action_logprobs(self, observation: str, action: str) -> tuple:
        """Get log probabilities for an action"""
        # Tokenize observation
        obs_encoding = self.tokenizer(
            observation,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.model.device)

        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**obs_encoding, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1][:, -1, :]  # Last token hidden state

            # Get value estimate
            value = self.value_net(hidden_states)

        # Tokenize action
        action_encoding = self.tokenizer(
            action,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.model.device)

        # Compute log probabilities
        action_ids = action_encoding.input_ids[:, 1:]  # Skip BOS token
        full_input = torch.cat([obs_encoding.input_ids, action_ids], dim=1)

        outputs = self.model(full_input)
        logits = outputs.logits[:, obs_encoding.input_ids.shape[1] - 1:-1, :]

        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = torch.gather(
            log_probs,
            dim=2,
            index=action_ids.unsqueeze(-1)
        ).squeeze(-1).sum(dim=-1)

        return action_log_probs, value, hidden_states

    def collect_rollouts(self, num_rollouts: int) -> RolloutBuffer:
        """Collect rollouts from the dataset"""
        self.rollout_buffer.clear()

        indices = np.random.choice(len(self.dataset), size=num_rollouts, replace=False)

        for idx in tqdm(indices, desc="Collecting rollouts"):
            sample = self.dataset[idx]

            # Extract observation and action from text
            text = sample["text"]
            obs_start = text.find("<|im_start|>user\n") + len("<|im_start|>user\n")
            obs_end = text.find("<|im_end|>", obs_start)
            observation = text[obs_start:obs_end]

            action_start = text.find("Action: ") + len("Action: ")
            action_end = text.find("<|im_end|>", action_start)
            action = text[action_start:action_end]

            # Get log probs and value
            log_prob, value, _ = self.get_action_logprobs(observation, action)

            # Store in buffer
            self.rollout_buffer.observations.append(observation)
            self.rollout_buffer.actions.append(action)
            self.rollout_buffer.log_probs.append(log_prob)
            self.rollout_buffer.values.append(value)
            self.rollout_buffer.rewards.append(sample["reward"])
            self.rollout_buffer.dones.append(sample["done"])

        # Compute advantages
        self.rollout_buffer.compute_advantages(self.gamma, self.gae_lambda)

        return self.rollout_buffer

    def ppo_update(self, buffer: RolloutBuffer, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """Perform PPO update"""
        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "total_loss": 0.0,
            "kl_div": 0.0,
        }

        for epoch in range(self.ppo_epochs):
            # Shuffle data
            indices = np.random.permutation(len(buffer.observations))

            for idx in indices:
                observation = buffer.observations[idx]
                action = buffer.actions[idx]
                old_log_prob = buffer.log_probs[idx]
                advantage = buffer.advantages[idx]
                return_val = buffer.returns[idx]

                # Get new log prob and value
                new_log_prob, value, _ = self.get_action_logprobs(observation, action)

                # Compute ratio
                ratio = torch.exp(new_log_prob - old_log_prob)

                # Clipped surrogate objective
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantage
                policy_loss = -torch.min(surr1, surr2)

                # Value loss
                value_loss = F.mse_loss(value, return_val)

                # Total loss
                loss = policy_loss + self.vf_coef * value_loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=1.0)
                optimizer.step()

                # Update metrics
                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["total_loss"] += loss.item()
                metrics["kl_div"] += (old_log_prob - new_log_prob).abs().item()

        # Average metrics
        num_updates = self.ppo_epochs * len(buffer.observations)
        for key in metrics:
            metrics[key] /= num_updates

        return metrics

    def train(self):
        """Run RL training"""
        print("\n" + "=" * 80)
        print("STARTING RL TRAINING")
        print("=" * 80 + "\n")

        # Setup optimizer
        params = list(self.model.parameters()) + list(self.value_net.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.rl_config.get("learning_rate", 1e-5))

        # Training loop
        num_iterations = self.num_rollouts
        best_reward = float('-inf')

        for iteration in range(num_iterations):
            print(f"\nIteration {iteration + 1}/{num_iterations}")

            # Collect rollouts
            buffer = self.collect_rollouts(self.rl_config.get("rollout_batch_size", 4))

            # Compute metrics
            avg_reward = np.mean(buffer.rewards)
            avg_return = np.mean([r.item() for r in buffer.returns])

            print(f"Average reward: {avg_reward:.4f}")
            print(f"Average return: {avg_return:.4f}")

            # PPO update
            update_metrics = self.ppo_update(buffer, optimizer)

            print(f"Policy loss: {update_metrics['policy_loss']:.4f}")
            print(f"Value loss: {update_metrics['value_loss']:.4f}")

            # Log to W&B
            if self.tracking_config.get("use_wandb", False):
                wandb.log({
                    "iteration": iteration,
                    "avg_reward": avg_reward,
                    "avg_return": avg_return,
                    **update_metrics
                })

            # Save best model
            if avg_reward > best_reward:
                best_reward = avg_reward
                save_dir = Path(self.rl_config.get("save_dir", "models")) / "rl_best"
                save_model(self.model, self.tokenizer, save_dir)
                print(f"Saved best model with reward: {best_reward:.4f}")

        # Save final model
        save_dir = Path(self.rl_config.get("save_dir", "models")) / "rl_final"
        save_model(self.model, self.tokenizer, save_dir, save_full_model=True)

        print("\n" + "=" * 80)
        print("RL TRAINING COMPLETED")
        print("=" * 80)

        return {"best_reward": best_reward}


def run_rl_training(config_path: Optional[str] = None):
    """Main function to run RL training"""
    import yaml

    # Load config
    if config_path is None:
        config_path = "config.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create trainer
    trainer = RLTrainer(config)

    # Setup
    trainer.setup()

    # Train
    metrics = trainer.train()

    return metrics


if __name__ == "__main__":
    run_rl_training()

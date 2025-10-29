"""Dataset wrappers for training"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from torch.utils.data import Dataset
from dataclasses import dataclass


@dataclass
class TrainingExample:
    """Training example with prompt and completion"""
    prompt: str
    completion: str
    environment: str
    metadata: Dict[str, Any]


class TrainingDataset(Dataset):
    """Dataset for supervised fine-tuning on Affine tasks"""

    def __init__(
        self,
        data_dir: Path,
        environments: List[str],
        tokenizer=None,
        max_length: int = 4096,
        include_metadata: bool = False
    ):
        self.data_dir = Path(data_dir)
        self.environments = environments
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_metadata = include_metadata
        self.examples = []

        self._load_data()

    def _load_data(self):
        """Load training data from disk"""
        for env in self.environments:
            env_name = env.replace(":", "_")
            data_file = self.data_dir / f"{env_name}_samples.jsonl"

            if not data_file.exists():
                print(f"Warning: {data_file} not found, skipping {env}")
                continue

            with open(data_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    self.examples.append(TrainingExample(
                        prompt=data["prompt"],
                        completion=data["response"],
                        environment=data["environment"],
                        metadata=data.get("metadata", {})
                    ))

        print(f"Loaded {len(self.examples)} training examples")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]

        # Format as chat
        text = f"<|im_start|>system\nYou are a helpful AI assistant specialized in reasoning and problem-solving.<|im_end|>\n<|im_start|>user\n{example.prompt}<|im_end|>\n<|im_start|>assistant\n{example.completion}<|im_end|>"

        if self.tokenizer is not None:
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )

            result = {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "labels": encoding["input_ids"].squeeze(),
            }
        else:
            result = {"text": text}

        if self.include_metadata:
            result["environment"] = example.environment
            result["metadata"] = example.metadata

        return result


class RLDataset(Dataset):
    """Dataset for RL training on AgentGym tasks"""

    def __init__(
        self,
        data_dir: Path,
        environments: List[str],
        tokenizer=None,
        max_length: int = 4096,
        format_type: str = "react"  # "react", "function_calling", "code"
    ):
        self.data_dir = Path(data_dir)
        self.environments = environments
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.format_type = format_type
        self.transitions = []

        self._load_data()

    def _load_data(self):
        """Load RL episodes from disk"""
        for env in self.environments:
            env_name = env.replace(":", "_")
            data_file = self.data_dir / f"{env_name}_episodes.jsonl"

            if not data_file.exists():
                print(f"Warning: {data_file} not found, skipping {env}")
                continue

            with open(data_file, 'r') as f:
                for line in f:
                    episode = json.loads(line)

                    # Extract transitions from episode
                    for i in range(len(episode["actions"])):
                        self.transitions.append({
                            "observation": episode["observations"][i],
                            "action": episode["actions"][i],
                            "reward": episode["rewards"][i],
                            "next_observation": episode["observations"][i + 1] if i + 1 < len(episode["observations"]) else "",
                            "done": episode["dones"][i],
                            "environment": episode["environment"]
                        })

        print(f"Loaded {len(self.transitions)} transitions from RL episodes")

    def _format_react(self, observation: str, action: str) -> str:
        """Format as REACT (Thought + Action)"""
        return f"""<|im_start|>system
You are an interactive agent. Respond with your thought process and then an action.
Format:
Thought: [your reasoning]
Action: [your action]<|im_end|>
<|im_start|>user
{observation}<|im_end|>
<|im_start|>assistant
Thought: Based on the observation, I should take the following action.
Action: {action}<|im_end|>"""

    def _format_function_calling(self, observation: str, action: str) -> str:
        """Format as function calling"""
        # Parse action to extract function and args
        if '[' in action and ']' in action:
            func_name = action[:action.index('[')]
            args = action[action.index('[') + 1:action.index(']')]
        else:
            func_name = action
            args = ""

        return f"""<|im_start|>system
You are an interactive agent. Call functions to interact with the environment.<|im_end|>
<|im_start|>user
{observation}<|im_end|>
<|im_start|>assistant
{{"function": "{func_name}", "arguments": "{args}"}}<|im_end|>"""

    def _format_code(self, observation: str, action: str) -> str:
        """Format as code-as-action"""
        return f"""<|im_start|>system
You are an interactive agent. Write Python code to take actions.<|im_end|>
<|im_start|>user
{observation}<|im_end|>
<|im_start|>assistant
```python
{action}
```<|im_end|>"""

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        transition = self.transitions[idx]

        # Format based on type
        if self.format_type == "react":
            text = self._format_react(transition["observation"], transition["action"])
        elif self.format_type == "function_calling":
            text = self._format_function_calling(transition["observation"], transition["action"])
        elif self.format_type == "code":
            text = self._format_code(transition["observation"], transition["action"])
        else:
            raise ValueError(f"Unknown format type: {self.format_type}")

        if self.tokenizer is not None:
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )

            return {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "labels": encoding["input_ids"].squeeze(),
                "reward": transition["reward"],
                "done": transition["done"],
                "environment": transition["environment"]
            }
        else:
            return {
                "text": text,
                "reward": transition["reward"],
                "done": transition["done"],
                "environment": transition["environment"]
            }


def create_train_val_split(
    data_dir: Path,
    val_ratio: float = 0.1,
    seed: int = 42
) -> tuple:
    """Split data into train and validation sets"""
    random.seed(seed)

    train_files = []
    val_files = []

    for file in data_dir.glob("*.jsonl"):
        with open(file, 'r') as f:
            lines = f.readlines()

        random.shuffle(lines)
        split_idx = int(len(lines) * (1 - val_ratio))

        train_data = lines[:split_idx]
        val_data = lines[split_idx:]

        # Save train split
        train_file = data_dir / f"train_{file.name}"
        with open(train_file, 'w') as f:
            f.writelines(train_data)
        train_files.append(train_file)

        # Save val split
        val_file = data_dir / f"val_{file.name}"
        with open(val_file, 'w') as f:
            f.writelines(val_data)
        val_files.append(val_file)

        print(f"Split {file.name}: {len(train_data)} train, {len(val_data)} val")

    return train_files, val_files

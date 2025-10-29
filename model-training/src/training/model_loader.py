"""Model loading utilities"""

import torch
from pathlib import Path
from typing import Optional, Tuple
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)


def load_model_and_tokenizer(
    model_path: str,
    use_lora: bool = True,
    lora_config: Optional[dict] = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    device_map: str = "auto",
    torch_dtype: str = "bfloat16",
    use_flash_attention: bool = True,
) -> Tuple:
    """
    Load model and tokenizer with optional LoRA and quantization

    Args:
        model_path: Path to the base model
        use_lora: Whether to use LoRA for efficient fine-tuning
        lora_config: LoRA configuration dict
        load_in_8bit: Whether to load model in 8-bit precision
        load_in_4bit: Whether to load model in 4-bit precision
        device_map: Device map for model placement
        torch_dtype: Torch dtype for model weights
        use_flash_attention: Whether to use flash attention 2

    Returns:
        (model, tokenizer) tuple
    """
    print(f"Loading model from {model_path}")

    # Convert torch_dtype string to actual dtype
    if torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    elif torch_dtype == "float16":
        dtype = torch.float16
    elif torch_dtype == "float32":
        dtype = torch.float32
    else:
        dtype = torch.bfloat16

    # Configure quantization if requested
    quantization_config = None
    if load_in_4bit or load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4" if load_in_4bit else None,
        )

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left"  # Important for batch generation
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    print("Loading model...")
    model_kwargs = {
        "pretrained_model_name_or_path": model_path,
        "torch_dtype": dtype,
        "device_map": device_map,
        "trust_remote_code": True,
    }

    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

    # Configure for training
    model.config.use_cache = False  # Disable KV cache for training

    # Apply LoRA if requested
    if use_lora:
        print("Applying LoRA...")

        # Prepare model for k-bit training if quantized
        if quantization_config is not None:
            model = prepare_model_for_kbit_training(model)

        # Default LoRA config
        default_lora_config = {
            "r": 64,
            "lora_alpha": 128,
            "lora_dropout": 0.05,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            "bias": "none",
            "task_type": TaskType.CAUSAL_LM,
        }

        # Override with provided config
        if lora_config is not None:
            default_lora_config.update(lora_config)

        # Convert target_modules list to format expected by LoraConfig
        if "target_modules" in default_lora_config:
            default_lora_config["target_modules"] = default_lora_config["target_modules"]

        peft_config = LoraConfig(**default_lora_config)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    print(f"Model loaded successfully!")
    print(f"Model dtype: {model.dtype}")
    print(f"Model device: {next(model.parameters()).device}")

    return model, tokenizer


def save_model(model, tokenizer, output_dir: Path, save_full_model: bool = False):
    """
    Save model and tokenizer

    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        output_dir: Output directory
        save_full_model: Whether to save full model (for LoRA, merge adapters first)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving model to {output_dir}")

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)

    # Save model
    if hasattr(model, 'save_pretrained'):
        if save_full_model and hasattr(model, 'merge_and_unload'):
            # Merge LoRA adapters and save full model
            print("Merging LoRA adapters...")
            model = model.merge_and_unload()

        model.save_pretrained(
            output_dir,
            safe_serialization=True,
            max_shard_size="5GB"
        )
    else:
        # If model is wrapped, unwrap it first
        unwrapped_model = model.module if hasattr(model, 'module') else model
        unwrapped_model.save_pretrained(
            output_dir,
            safe_serialization=True,
            max_shard_size="5GB"
        )

    print(f"Model saved successfully!")


def count_parameters(model) -> dict:
    """Count trainable and total parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_percentage": 100 * trainable_params / total_params if total_params > 0 else 0
    }

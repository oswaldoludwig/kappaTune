# kappatune/hf_integration.py
from typing import Dict, List, Optional
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from kappaTune.selective_fine_tuning import SelectiveFineTuningOptimizer

def get_kappa_lora_config(
    model: nn.Module,
    num_modules_to_adapt: int = 50,          # budget of low-κ modules
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,  # override if you want manual
    task_type: TaskType = TaskType.CAUSAL_LM,
    **lora_kwargs
) -> LoraConfig:
    """
    Computes low-κ modules (KappaTune selection) and returns a LoraConfig
    that applies LoRA *only* to those modules → KappaTune-LoRA.
    """
    if target_modules is None:
        # Reuse your analyzer (temporarily set to 0 trainable so we only get selection)
        analyzer = SelectiveFineTuningOptimizer(
            model=model,
            base_optimizer_cls=torch.optim.AdamW,  # dummy
            optimizer_args={"lr": 1e-5},
            num_tensors_to_finetune=num_modules_to_adapt * 2,  # over-select tensors → modules
            recompute=True,
            max_dim_size_to_analyze=16384,  # safety for embeddings
        )
        
        # Map selected param names → parent module names
        selected_modules = set()
        for param_name in analyzer.trainable_param_names:
            # e.g. "model.layers.0.self_attn.q_proj.weight" → "model.layers.0.self_attn.q_proj"
            module_name = ".".join(param_name.split(".")[:-1])  # drop .weight / .bias
            selected_modules.add(module_name)
        
        target_modules = list(selected_modules)[:num_modules_to_adapt]  # final budget

    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        task_type=task_type,
        **lora_kwargs
    )
    return config


# Convenience wrapper (mimics PEFT API)
def get_kappatune_lora_model(
    model: nn.Module,
    num_modules_to_adapt: int = 50,
    **lora_kwargs
):
    config = get_kappa_lora_config(model, num_modules_to_adapt, **lora_kwargs)
    return get_peft_model(model, config)

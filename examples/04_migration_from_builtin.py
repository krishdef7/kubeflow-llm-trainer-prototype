#!/usr/bin/env python3
# Copyright 2026 The Kubeflow Authors.
# Licensed under the Apache License, Version 2.0.

"""Example 4: Migration from BuiltinTrainer to LLMTrainer.

Shows the zero-change migration path: existing BuiltinTrainer code
continues to work, transparently routed through the new framework.
"""

from dataclasses import dataclass
from typing import Any

from kubeflow_llm_trainer import BackendRegistry, FineTuningMethod, LLMConfig, LLMTrainer
from kubeflow_llm_trainer._compat import adapt_builtin_trainer
from kubeflow_llm_trainer.backends.torchtune import TorchTuneBackend
from kubeflow_llm_trainer.backends.trl import TRLBackend

BackendRegistry.register(TorchTuneBackend())


# --- Simulate existing SDK types ---

@dataclass
class LoraConfig:
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    quantize_base: bool = False


@dataclass
class TorchTuneConfig:
    model_args: str | None = None
    dataset_args: str | None = None
    dtype: str | None = "bf16"
    batch_size: int | None = 4
    epochs: int | None = 3
    loss: Any = None
    peft_config: LoraConfig | None = None
    dataset_preprocess_config: Any = None
    resources_per_node: dict[str, Any] | None = None
    num_nodes: int | None = 1


@dataclass
class BuiltinTrainer:
    config: TorchTuneConfig


# =========================================================================
# BEFORE: Existing code (works today, continues to work)
# =========================================================================

print("=== BEFORE (existing BuiltinTrainer) ===")
old_trainer = BuiltinTrainer(
    config=TorchTuneConfig(
        # NOTE: model_args is a TorchTune recipe identifier, not a HuggingFace
        # model ID.  The adapter passes it through as LLMConfig.model — this is
        # a lossy mapping.  Users migrating directly should use the HuggingFace
        # model ID and set extra_args["torchtune_config"] for the base config.
        model_args="llama3_2/1B_lora_single_device",
        dataset_args="alpaca_dataset",
        peft_config=LoraConfig(lora_rank=8, lora_alpha=16),
        epochs=3,
        batch_size=4,
        resources_per_node={"gpu": 1},
    )
)
# In TrainerClient.train(), this is transparently converted:
new_trainer = adapt_builtin_trainer(old_trainer)
resolved = new_trainer.resolve()
print(f"  Backend:  {resolved.backend_name}")
print(f"  Method:   {new_trainer.config.method.value}")
print(f"  Command:  {' '.join(resolved.container_spec.command)}")

# =========================================================================
# AFTER: New code (same result, more flexible)
# =========================================================================

print("\n=== AFTER (new LLMTrainer) ===")
new_trainer = LLMTrainer(
    config=LLMConfig(
        model="meta-llama/Llama-3.2-1B-Instruct",
        dataset="tatsu-lab/alpaca",
        method=FineTuningMethod.LORA,
        backend_name="torchtune",
        lora_config={"r": 8, "lora_alpha": 16},
        epochs=3,
        batch_size=4,
    ),
    resources_per_node={"gpu": 1},
)
resolved = new_trainer.resolve()
print(f"  Backend:  {resolved.backend_name}")
print(f"  Method:   {new_trainer.config.method.value}")
print(f"  Command:  {' '.join(resolved.container_spec.command)}")

# =========================================================================
# BONUS: Switch to TRL — one line change
# =========================================================================

print("\n=== BONUS: Switch to TRL (one line change) ===")
BackendRegistry.register(TRLBackend())

trl_trainer = LLMTrainer(
    config=LLMConfig(
        model="meta-llama/Llama-3.2-1B-Instruct",
        dataset="tatsu-lab/alpaca",
        method=FineTuningMethod.SFT,           # SFT with LoRA
        backend_name="trl",                     # ← only this changed
        lora_config={"r": 8, "lora_alpha": 16},
        epochs=3,
        batch_size=4,
    ),
    resources_per_node={"gpu": 1},
)
resolved = trl_trainer.resolve()
print(f"  Backend:  {resolved.backend_name}")
print(f"  Method:   {trl_trainer.config.method.value}")
print(f"  Command:  {' '.join(resolved.container_spec.command)}")

#!/usr/bin/env python3
# Copyright 2026 The Kubeflow Authors.
# Licensed under the Apache License, Version 2.0.

"""Example 2: Cross-backend switching — same model, different engines.

This is the KEY VALUE PROPOSITION of the Dynamic LLM Trainer Framework:
users can switch between TorchTune, TRL, and Unsloth by changing one field.
"""

from kubeflow_llm_trainer import (
    BackendRegistry,
    FineTuningMethod,
    LLMConfig,
    LLMTrainer,
)
from kubeflow_llm_trainer.backends.torchtune import TorchTuneBackend
from kubeflow_llm_trainer.backends.trl import TRLBackend
from kubeflow_llm_trainer.backends.unsloth import UnslothBackend

BackendRegistry.register(TorchTuneBackend())
BackendRegistry.register(TRLBackend())
BackendRegistry.register(UnslothBackend())

# Same model, same dataset, same LoRA config — three backends.
base_kwargs = dict(
    model="meta-llama/Llama-3.2-1B-Instruct",
    dataset="tatsu-lab/alpaca",
    method=FineTuningMethod.SFT,
    lora_config={"r": 16, "lora_alpha": 32},
    epochs=3,
    batch_size=4,
)

for backend_name in ["torchtune", "trl", "unsloth"]:
    # Adjust method for TRL/Unsloth (they use SFT + lora_config, not LORA method)
    method = FineTuningMethod.SFT if backend_name in ("trl", "unsloth") else FineTuningMethod.LORA
    config = LLMConfig(**{**base_kwargs, "method": method, "backend_name": backend_name})
    trainer = LLMTrainer(config=config, resources_per_node={"gpu": 1})
    resolved = trainer.resolve()

    print(f"\n{'='*60}")
    print(f"Backend: {resolved.backend_name}")
    print(f"  Image:    {resolved.container_spec.image}")
    print(f"  Command:  {' '.join(resolved.container_spec.command)}")
    print(f"  Packages: {resolved.packages_to_install or '(none — built into image)'}")

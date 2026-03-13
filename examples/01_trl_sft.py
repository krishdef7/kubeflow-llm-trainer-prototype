#!/usr/bin/env python3
# Copyright 2026 The Kubeflow Authors.
# Licensed under the Apache License, Version 2.0.

"""Example 1: SFT fine-tuning with TRL backend.

Shows how a user would fine-tune Llama 3.2 with supervised learning
using the TRL backend — the most common post-training workflow.
"""

from kubeflow_llm_trainer import (
    BackendRegistry,
    FineTuningMethod,
    LLMConfig,
    LLMTrainer,
)
from kubeflow_llm_trainer.backends.trl import TRLBackend
from kubeflow_llm_trainer.integration import build_trainjob_spec

import json

# In the real SDK, backends are auto-discovered via entry points.
BackendRegistry.register(TRLBackend())

# --- User code starts here ---

trainer = LLMTrainer(
    config=LLMConfig(
        model="meta-llama/Llama-3.2-1B-Instruct",
        dataset="tatsu-lab/alpaca",
        method=FineTuningMethod.SFT,
        backend_name="trl",
        lora_config={
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
        epochs=3,
        batch_size=4,
        learning_rate=2e-5,
    ),
    resources_per_node={"gpu": 2, "memory": "64Gi"},
)

# In the real SDK: client.train(name="llama-sft", trainer=trainer)
# Here we just show the resolved spec.
resolved = trainer.resolve()
spec = build_trainjob_spec("llama-sft", resolved, runtime_name="torch-distributed")

print("=== Resolved LLM Trainer ===")
print(f"Backend:    {resolved.backend_name}")
print(f"Framework:  {resolved.framework}")
print(f"Image:      {resolved.container_spec.image}")
print(f"Command:    {resolved.container_spec.command}")
print(f"Args:       {resolved.container_spec.args}")
print(f"Packages:   {resolved.packages_to_install}")
print()
print("=== TrainJob Spec (YAML-equivalent) ===")
print(json.dumps(spec.to_dict(), indent=2))

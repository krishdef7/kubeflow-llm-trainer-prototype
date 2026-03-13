#!/usr/bin/env python3
# Copyright 2026 The Kubeflow Authors.
# Licensed under the Apache License, Version 2.0.

"""Example 3: External backend registration.

Shows how a third-party package can register a custom LLM backend.
This mirrors how TrainingRuntime enables custom runtimes on the
controller side.

To distribute as a pip package, add to pyproject.toml:

    [project.entry-points."kubeflow.llm_backends"]
    llama_factory = "my_package.backends:LlamaFactoryBackend"
"""

from __future__ import annotations

import json

from kubeflow_llm_trainer import (
    BackendRegistry,
    ContainerSpec,
    FineTuningMethod,
    LLMBackend,
    LLMConfig,
    LLMTrainer,
)


@BackendRegistry.register
class LlamaFactoryBackend(LLMBackend):
    """Custom backend for LlamaFactory — 100+ model support.

    LlamaFactory provides a unified interface for fine-tuning 100+ LLMs
    with various methods.  This example shows how it integrates as an
    external backend.
    """

    @property
    def name(self) -> str:
        return "llama_factory"

    @property
    def packages_to_install(self) -> list[str]:
        return ["llamafactory>=0.9.0"]

    @property
    def supported_methods(self) -> list[FineTuningMethod]:
        return [
            FineTuningMethod.SFT,
            FineTuningMethod.DPO,
            FineTuningMethod.PPO,
            FineTuningMethod.ORPO,
            FineTuningMethod.KTO,
            FineTuningMethod.LORA,
            FineTuningMethod.QLORA,
        ]

    def validate(self, config: LLMConfig) -> None:
        # LlamaFactory supports virtually everything.
        pass

    def to_container_spec(self, config: LLMConfig) -> ContainerSpec:
        lf_config = {
            "model_name_or_path": config.model,
            "dataset": config.dataset,
            "stage": self._map_stage(config.method),
            "finetuning_type": "lora" if config.is_peft else "full",
            "num_train_epochs": config.epochs,
            "per_device_train_batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "output_dir": "/mnt/output",
        }

        if config.lora_config:
            lf_config["lora_rank"] = config.lora_config.get("r", 8)
            lf_config["lora_alpha"] = config.lora_config.get("lora_alpha", 16)

        return ContainerSpec(
            image="ghcr.io/my-org/llama-factory-trainer:latest",
            command=["llamafactory-cli", "train"],
            args=["--config_json", json.dumps(lf_config)],
            env={
                "KUBEFLOW_MODEL_PATH": "/mnt/model",
                "KUBEFLOW_DATASET_PATH": "/mnt/dataset",
            },
        )

    @staticmethod
    def _map_stage(method: FineTuningMethod) -> str:
        mapping = {
            FineTuningMethod.SFT: "sft",
            FineTuningMethod.DPO: "dpo",
            FineTuningMethod.PPO: "ppo",
            FineTuningMethod.ORPO: "orpo",
            FineTuningMethod.KTO: "kto",
            FineTuningMethod.LORA: "sft",
            FineTuningMethod.QLORA: "sft",
            FineTuningMethod.FULL: "sft",
        }
        return mapping.get(method, "sft")


# --- Use the custom backend ---

trainer = LLMTrainer(
    config=LLMConfig(
        model="meta-llama/Llama-3.2-1B-Instruct",
        dataset="tatsu-lab/alpaca",
        method=FineTuningMethod.SFT,
        backend_name="llama_factory",
        lora_config={"r": 16, "lora_alpha": 32},
    ),
    resources_per_node={"gpu": 2},
)

resolved = trainer.resolve()
print(f"Backend:  {resolved.backend_name}")
print(f"Image:    {resolved.container_spec.image}")
print(f"Command:  {resolved.container_spec.command}")
print(f"Args:     {resolved.container_spec.args}")
print(f"\nRegistered backends: {BackendRegistry.list_backends()}")

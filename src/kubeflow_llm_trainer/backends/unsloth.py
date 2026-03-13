# Copyright 2026 The Kubeflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unsloth backend — ~2× faster, ~70% lower memory LLM fine-tuning.

Unsloth achieves dramatic speedups by:

* Custom CUDA kernels for attention and cross-entropy
* Intelligent memory management with gradient checkpointing
* Optimized LoRA weight merging

Unsloth wraps the Hugging Face Trainer / TRL API, so the training script
structure is similar to TRL, but uses ``unsloth.FastLanguageModel`` for
model loading and patching.

The entrypoint runs a generated training script that:

1. Loads the model with ``FastLanguageModel.from_pretrained(...)``
2. Applies LoRA with ``FastLanguageModel.get_peft_model(...)``
3. Trains with ``SFTTrainer`` / ``DPOTrainer`` from TRL (Unsloth patches these)
4. Saves the model

Limitations:
- Only LoRA/QLoRA — no full fine-tuning
- Single-GPU only (no distributed training)
- Supports a curated list of models (Llama, Mistral, Gemma, Phi, Qwen, etc.)
"""

from __future__ import annotations

import json

from kubeflow_llm_trainer.interface import (
    ContainerSpec,
    DataType,
    FineTuningMethod,
    LLMBackend,
    LLMConfig,
)

_DEFAULT_IMAGE = "ghcr.io/kubeflow/trainer/unsloth-trainer:latest"

_HF_DTYPE_MAP = {
    DataType.FP32: "float32",
    DataType.FP16: "float16",
    DataType.BF16: "bfloat16",
}


class UnslothBackend(LLMBackend):
    """Pluggable backend for Unsloth-accelerated fine-tuning.

    Unsloth provides ~2× training speedup and ~70% lower memory usage
    compared to standard Hugging Face training, with zero accuracy loss.
    """

    @property
    def name(self) -> str:
        return "unsloth"

    @property
    def packages_to_install(self) -> list[str]:
        # NOTE: Do NOT use unsloth[colab-new] — that extra runs Colab-specific
        # pip tricks that conflict with a properly configured CUDA container.
        # In production, the base image should have the correct CUDA runtime.
        return ["unsloth>=2024.12", "trl>=0.12.0", "peft>=0.13.0"]

    @property
    def supported_methods(self) -> list[FineTuningMethod]:
        return [
            FineTuningMethod.SFT,
            FineTuningMethod.DPO,
            FineTuningMethod.ORPO,
            FineTuningMethod.LORA,
            FineTuningMethod.QLORA,
        ]

    def validate(self, config: LLMConfig) -> None:
        """Validate Unsloth-specific constraints."""
        # Unsloth requires LoRA — it patches the model for LoRA-specific
        # kernel optimizations.
        if config.method == FineTuningMethod.FULL:
            raise ValueError(
                "Unsloth does not support full fine-tuning.  "
                "It requires LoRA/QLoRA for its optimized kernels to work.  "
                "Use method=FineTuningMethod.LORA or FineTuningMethod.SFT "
                "with lora_config set."
            )
        if config.method == FineTuningMethod.PPO:
            raise ValueError(
                "Unsloth does not support PPO.  Use the 'trl' backend."
            )
        if config.method == FineTuningMethod.KTO:
            raise ValueError(
                "Unsloth does not support KTO.  Use the 'trl' backend."
            )

        # Unsloth is single-GPU only.
        if config.num_nodes > 1:
            raise ValueError(
                f"Unsloth supports single-GPU training only (got num_nodes="
                f"{config.num_nodes}).  For multi-node training, use 'trl' or "
                f"'torchtune' backend."
            )

        # Validate LoRA config if provided.
        if config.lora_config:
            _allowed_keys = {
                "r", "lora_alpha", "lora_dropout", "target_modules",
                "bias", "use_gradient_checkpointing", "random_state",
                "max_seq_length", "use_rslora",
            }
            unknown = set(config.lora_config) - _allowed_keys
            if unknown:
                raise ValueError(
                    f"Unknown Unsloth LoRA config keys: {unknown}. "
                    f"Allowed: {sorted(_allowed_keys)}."
                )

    def to_container_spec(self, config: LLMConfig) -> ContainerSpec:
        """Generate the Unsloth training script arguments."""
        training_config = self._build_config(config)
        config_json = json.dumps(training_config, indent=2)

        # Determine the trainer type based on the method.
        trainer_type = "sft"
        if config.method == FineTuningMethod.DPO:
            trainer_type = "dpo"
        elif config.method == FineTuningMethod.ORPO:
            trainer_type = "orpo"

        env: dict[str, str] = {
            "UNSLOTH_TRAINING_CONFIG": config_json,
            "UNSLOTH_TRAINER_TYPE": trainer_type,
            "KUBEFLOW_MODEL_PATH": "/mnt/model",
            "KUBEFLOW_DATASET_PATH": "/mnt/dataset",
        }

        return ContainerSpec(
            image=config.extra_args.get("image", _DEFAULT_IMAGE),
            command=[
                "python", "-m",
                "kubeflow_llm_trainer.entrypoints.unsloth_runner",
            ],
            args=[],
            env=env,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_config(self, config: LLMConfig) -> dict:
        """Build the Unsloth training configuration."""
        lora = config.lora_config or {}

        return {
            "model_name": config.model,
            "dataset_name": config.dataset,
            "dtype": _HF_DTYPE_MAP.get(config.dtype, "bfloat16"),
            "max_seq_length": lora.get("max_seq_length", 2048),
            "load_in_4bit": config.method == FineTuningMethod.QLORA,
            "lora_r": lora.get("r", 16),
            "lora_alpha": lora.get("lora_alpha", 16),
            "lora_dropout": lora.get("lora_dropout", 0.0),
            "target_modules": lora.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]),
            "use_gradient_checkpointing": lora.get(
                "use_gradient_checkpointing", "unsloth"
            ),
            "use_rslora": lora.get("use_rslora", False),
            "random_state": lora.get("random_state", 3407),
            "num_train_epochs": config.epochs,
            "per_device_train_batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "output_dir": "/mnt/output",
        }

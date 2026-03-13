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

"""TorchTune backend — preserves existing functionality as a pluggable backend.

This backend wraps TorchTune and produces a ``ContainerSpec`` equivalent to what
the current ``BuiltinTrainer + TorchTuneConfig`` path generates.  It is the
*default* backend, ensuring full backward compatibility.

TorchTune CLI format
--------------------

TorchTune uses the ``tune run`` CLI with YAML configs and OmegaConf-style
key=value overrides::

    tune run <recipe> --config <base_config> key1=value1 key2=value2

For example::

    tune run lora_finetune_single_device \\
        --config llama3_2/1B_lora_single_device \\
        epochs=3 batch_size=4 dtype=bf16

See: https://pytorch.org/torchtune/stable/deep_dives/configs.html

LoRA parameter mapping
----------------------

TorchTune's recipe configs use **top-level** keys for LoRA parameters,
not a nested ``lora.`` namespace::

    # In the TorchTune YAML config:
    lora_rank: 8
    lora_alpha: 16
    lora_dropout: 0.1
    lora_attn_modules: [q_proj, k_proj, v_proj]

So the CLI overrides are ``lora_rank=8``, not ``lora.rank=8``.

The ``LLMConfig.lora_config`` dict uses short keys (``r``, ``lora_alpha``,
etc.) which are mapped to TorchTune's top-level keys via ``_LORA_KEY_MAP``.
"""

from __future__ import annotations

from kubeflow_llm_trainer.interface import (
    ContainerSpec,
    DataType,
    FineTuningMethod,
    LLMBackend,
    LLMConfig,
)

# Default TorchTune trainer image from the Kubeflow manifests.
_DEFAULT_IMAGE = "ghcr.io/kubeflow/trainer/torchtune-trainer:latest"

# TorchTune recipe names indexed by (method, is_distributed).
# Ref: `tune ls` output from torchtune >= 0.4
_RECIPES: dict[tuple[FineTuningMethod, bool], str] = {
    (FineTuningMethod.FULL, False): "full_finetune_single_device",
    (FineTuningMethod.FULL, True): "full_finetune_distributed",
    (FineTuningMethod.SFT, False): "full_finetune_single_device",
    (FineTuningMethod.SFT, True): "full_finetune_distributed",
    (FineTuningMethod.LORA, False): "lora_finetune_single_device",
    (FineTuningMethod.LORA, True): "lora_finetune_distributed",
    # QLoRA uses the same lora recipes with quantize_base=True override.
    # torchtune >= 0.5 also provides dedicated qlora_* recipes.
    (FineTuningMethod.QLORA, False): "qlora_finetune_single_device",
    (FineTuningMethod.QLORA, True): "qlora_finetune_distributed",
    (FineTuningMethod.DPO, False): "lora_dpo_single_device",
    (FineTuningMethod.DPO, True): "lora_dpo_distributed",
}

_DTYPE_MAP: dict[DataType, str] = {
    DataType.FP32: "fp32",
    DataType.FP16: "fp16",
    DataType.BF16: "bf16",
}

# Maps LLMConfig.lora_config keys → TorchTune top-level config keys.
# TorchTune uses top-level keys (lora_rank, lora_alpha, etc.), NOT a nested
# `lora.` namespace.  See any lora_finetune_* recipe config YAML.
_LORA_KEY_MAP: dict[str, str] = {
    "r": "lora_rank",
    "lora_alpha": "lora_alpha",
    "lora_dropout": "lora_dropout",
    "lora_attn_modules": "lora_attn_modules",
    "apply_lora_to_mlp": "apply_lora_to_mlp",
    "apply_lora_to_output": "apply_lora_to_output",
    "quantize_base": "quantize_base",
}


class TorchTuneBackend(LLMBackend):
    """Pluggable backend for TorchTune-based fine-tuning.

    This is the refactored version of the current ``BuiltinTrainer`` +
    ``TorchTuneConfig`` path.  It produces the same container commands
    that the existing ``ClusterTrainingRuntime`` for TorchTune expects.
    """

    @property
    def name(self) -> str:
        return "torchtune"

    @property
    def supported_methods(self) -> list[FineTuningMethod]:
        return [
            FineTuningMethod.SFT,
            FineTuningMethod.FULL,
            FineTuningMethod.LORA,
            FineTuningMethod.QLORA,
            FineTuningMethod.DPO,
        ]

    def validate(self, config: LLMConfig) -> None:
        """Validate TorchTune-specific constraints."""
        if config.method == FineTuningMethod.PPO:
            raise ValueError(
                "TorchTune does not support PPO training.  "
                "Use the 'trl' backend for PPO: "
                "LLMConfig(backend_name='trl', method=FineTuningMethod.PPO, ...)"
            )
        if config.method == FineTuningMethod.ORPO:
            raise ValueError(
                "TorchTune does not support ORPO.  Use the 'trl' backend."
            )
        if config.method == FineTuningMethod.KTO:
            raise ValueError(
                "TorchTune does not support KTO.  Use the 'trl' backend."
            )
        if config.lora_config:
            unknown = set(config.lora_config) - set(_LORA_KEY_MAP)
            if unknown:
                raise ValueError(
                    f"Unknown TorchTune LoRA config keys: {unknown}. "
                    f"Allowed keys: {sorted(_LORA_KEY_MAP)}."
                )

    def to_container_spec(self, config: LLMConfig) -> ContainerSpec:
        """Generate a ``tune run`` command from the LLMConfig.

        TorchTune CLI uses OmegaConf-style ``key=value`` overrides::

            tune run <recipe> --config <base_config> epochs=3 batch_size=4

        See: https://pytorch.org/torchtune/stable/deep_dives/configs.html
        """
        recipe_key = (config.method, config.is_distributed)
        recipe = _RECIPES.get(recipe_key)
        if recipe is None:
            raise ValueError(
                f"No TorchTune recipe for method={config.method.value}, "
                f"distributed={config.is_distributed}."
            )

        # The base config determines the TorchTune YAML config to load.
        # In production, the ClusterTrainingRuntime passes a model-specific
        # config name via extra_args["torchtune_config"], e.g.:
        #   extra_args={"torchtune_config": "llama3_2/1B_lora_single_device"}
        #
        # When no override is given, we pass the recipe name as the config.
        # This is valid TorchTune behavior: `tune run` resolves built-in
        # recipe configs by name, so:
        #   tune run lora_finetune_single_device --config lora_finetune_single_device
        # loads the default config for that recipe.
        base_config = config.extra_args.get("torchtune_config", recipe)

        # Build CLI args: tune run <recipe> --config <base> key=value ...
        args: list[str] = [recipe, "--config", base_config]

        # OmegaConf-style key=value overrides for training hyperparameters.
        args.append(f"epochs={config.epochs}")
        args.append(f"batch_size={config.batch_size}")
        args.append(f"dtype={_DTYPE_MAP[config.dtype]}")
        args.append(f"optimizer.lr={config.learning_rate}")

        # Point TorchTune's checkpointer at the Kubeflow model/output PVC mounts.
        # The Kubeflow init container downloads the model to /mnt/model; tune run
        # doesn't read KUBEFLOW_MODEL_PATH from the environment, so we inject the
        # paths as OmegaConf overrides that the checkpointer config consumes.
        args.append("checkpointer.checkpoint_dir=/mnt/model")
        args.append("checkpointer.output_dir=/mnt/output")

        # LoRA overrides — mapped from LLMConfig short keys to TorchTune's
        # top-level config keys (see _LORA_KEY_MAP).
        if config.lora_config:
            for key, value in config.lora_config.items():
                torchtune_key = _LORA_KEY_MAP.get(key, key)
                if isinstance(value, bool):
                    args.append(f"{torchtune_key}={str(value).lower()}")
                elif isinstance(value, list):
                    items = ",".join(str(v) for v in value)
                    args.append(f"{torchtune_key}=[{items}]")
                else:
                    args.append(f"{torchtune_key}={value}")

        # Pass-through extra args as OmegaConf overrides.
        for key, value in config.extra_args.items():
            if key in ("torchtune_config", "image"):
                continue  # handled separately
            args.append(f"{key}={value}")

        env: dict[str, str] = {
            "KUBEFLOW_MODEL_PATH": "/mnt/model",
            "KUBEFLOW_DATASET_PATH": "/mnt/dataset",
        }

        return ContainerSpec(
            image=config.extra_args.get("image", _DEFAULT_IMAGE),
            command=["tune", "run"],
            args=args,
            env=env,
        )

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

"""TRL backend — Hugging Face TRL for post-training methods.

TRL (Transformer Reinforcement Learning) is the industry standard for
post-training LLM alignment:

* **SFT**  — Supervised Fine-Tuning via ``SFTTrainer``
* **DPO**  — Direct Preference Optimization via ``DPOTrainer``
* **PPO**  — Proximal Policy Optimization via ``PPOTrainer``
* **ORPO** — Odds Ratio Preference Optimization via ``ORPOTrainer``
* **KTO**  — Kahneman-Tversky Optimization via ``KTOTrainer``

This is the primary new backend that motivates the Dynamic LLM Trainer
Framework.  TorchTune does not support PPO/ORPO/KTO, and TRL covers all
five methods with a unified API.

Container entrypoint
--------------------

TRL ships a CLI since v0.9.  The entrypoint writes training args to a temp
JSON file and invokes::

    trl <command> --config /tmp/training_args.json

Where ``<command>`` is one of ``sft``, ``dpo``, ``ppo``, ``orpo``, ``kto``.

TrainingRuntime mapping
-----------------------

This backend expects a ``ClusterTrainingRuntime`` with::

    metadata:
      labels:
        trainer.kubeflow.org/framework: torch
    spec:
      template:
        spec:
          replicatedJobs:
            - template:
                spec:
                  template:
                    spec:
                      containers:
                        - name: trainer
                          image: ghcr.io/kubeflow/trainer/trl-trainer:latest
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

_DEFAULT_IMAGE = "ghcr.io/kubeflow/trainer/trl-trainer:latest"

# TRL CLI subcommand for each method.
_TRL_COMMANDS: dict[FineTuningMethod, str] = {
    FineTuningMethod.SFT: "sft",
    FineTuningMethod.DPO: "dpo",
    FineTuningMethod.PPO: "ppo",
    FineTuningMethod.ORPO: "orpo",
    FineTuningMethod.KTO: "kto",
}

# Torch dtype strings for Hugging Face.
_HF_DTYPE_MAP: dict[DataType, str] = {
    DataType.FP32: "float32",
    DataType.FP16: "float16",
    DataType.BF16: "bfloat16",
}


class TRLBackend(LLMBackend):
    """Pluggable backend for Hugging Face TRL fine-tuning.

    Supports SFT, DPO, PPO, ORPO, and KTO — the full spectrum of
    post-training alignment methods.
    """

    @property
    def name(self) -> str:
        return "trl"

    @property
    def packages_to_install(self) -> list[str]:
        return ["trl>=0.12.0", "peft>=0.13.0", "accelerate>=1.0.0"]

    @property
    def supported_methods(self) -> list[FineTuningMethod]:
        # NOTE: LORA and QLORA are deliberately excluded here.  TRL treats
        # LoRA as a *modifier* applied on top of SFT/DPO/etc (via PEFT),
        # not as a standalone fine-tuning method.  The correct TRL usage is
        # method=SFT + lora_config={...}, not method=LORA.
        #
        # validate() catches LORA/QLORA with a helpful redirect message
        # *before* the supported_methods check in LLMTrainer.resolve() runs,
        # so users always get actionable guidance.
        return [
            FineTuningMethod.SFT,
            FineTuningMethod.DPO,
            FineTuningMethod.PPO,
            FineTuningMethod.ORPO,
            FineTuningMethod.KTO,
        ]

    def validate(self, config: LLMConfig) -> None:
        """Validate TRL-specific constraints."""
        if config.method == FineTuningMethod.FULL:
            raise ValueError(
                "TRL does not have a 'full' fine-tuning mode.  "
                "Use method=FineTuningMethod.SFT for supervised fine-tuning, "
                "or use the 'torchtune' backend for full-parameter training."
            )

        if config.method == FineTuningMethod.QLORA:
            raise ValueError(
                "For QLoRA with TRL, use method=FineTuningMethod.SFT (or DPO, "
                "etc.) and set lora_config with quantize_base=True. TRL handles "
                "quantization through BitsAndBytesConfig, not a separate method."
            )

        if config.method == FineTuningMethod.LORA:
            raise ValueError(
                "For LoRA with TRL, use method=FineTuningMethod.SFT (or DPO, "
                "etc.) and set lora_config. TRL applies LoRA via PEFT "
                "automatically when a lora_config is provided."
            )

        # PPO requires a reward model — validate upfront, not at spec time.
        if config.method == FineTuningMethod.PPO:
            if not config.extra_args.get("reward_model"):
                raise ValueError(
                    "PPO training requires 'reward_model' in extra_args, e.g.: "
                    "LLMConfig(method=FineTuningMethod.PPO, "
                    "extra_args={'reward_model': 'OpenAssistant/reward-model-deberta-v3'})"
                )

        # Validate LoRA config keys.
        if config.lora_config:
            _allowed_keys = {
                "r", "lora_alpha", "lora_dropout", "target_modules",
                "bias", "task_type", "modules_to_save", "quantize_base",
            }
            unknown = set(config.lora_config) - _allowed_keys
            if unknown:
                raise ValueError(
                    f"Unknown TRL/PEFT LoRA config keys: {unknown}. "
                    f"Allowed: {sorted(_allowed_keys)}."
                )

    def to_container_spec(self, config: LLMConfig) -> ContainerSpec:
        """Generate the TRL CLI command and training args."""
        trl_command = _TRL_COMMANDS.get(config.method)
        if trl_command is None:
            raise ValueError(
                f"No TRL command for method {config.method.value!r}."
            )

        # Build the training arguments dict that will be serialized to JSON.
        training_args = self._build_training_args(config)

        # Build PEFT config if LoRA is requested.
        peft_args = self._build_peft_args(config) if config.lora_config else {}

        # Merge all args into a single JSON payload.
        all_args = {**training_args, **peft_args}
        args_json = json.dumps(all_args, indent=2)

        # The TRL CLI reads from a YAML/JSON config file.
        # We pass the serialized args via an environment variable that the
        # entrypoint script writes to a temp file before invoking TRL.
        env: dict[str, str] = {
            "TRL_TRAINING_ARGS": args_json,
            "KUBEFLOW_MODEL_PATH": "/mnt/model",
            "KUBEFLOW_DATASET_PATH": "/mnt/dataset",
        }

        return ContainerSpec(
            image=config.extra_args.get("image", _DEFAULT_IMAGE),
            command=["python", "-m", "kubeflow_llm_trainer.entrypoints.trl_runner"],
            args=[trl_command],
            env=env,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_training_args(self, config: LLMConfig) -> dict:
        """Build Hugging Face TrainingArguments."""
        args: dict = {
            "model_name_or_path": config.model,
            "dataset_name": config.dataset,
            "num_train_epochs": config.epochs,
            "per_device_train_batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "bf16": config.dtype == DataType.BF16,
            "fp16": config.dtype == DataType.FP16,
            "output_dir": "/mnt/output",
            "logging_steps": config.extra_args.get("logging_steps", 10),
            "save_strategy": config.extra_args.get("save_strategy", "epoch"),
            "gradient_accumulation_steps": config.extra_args.get(
                "gradient_accumulation_steps", 1
            ),
        }

        # Method-specific defaults.
        if config.method == FineTuningMethod.DPO:
            args["beta"] = config.extra_args.get("beta", 0.1)
        elif config.method == FineTuningMethod.PPO:
            args["reward_model"] = config.extra_args["reward_model"]
        elif config.method == FineTuningMethod.KTO:
            args["desirable_weight"] = config.extra_args.get(
                "desirable_weight", 1.0
            )
            args["undesirable_weight"] = config.extra_args.get(
                "undesirable_weight", 1.0
            )

        return args

    def _build_peft_args(self, config: LLMConfig) -> dict:
        """Build PEFT / LoRA arguments."""
        if not config.lora_config:
            return {}

        peft = {
            "use_peft": True,
            "lora_r": config.lora_config.get("r", 16),
            "lora_alpha": config.lora_config.get("lora_alpha", 32),
            "lora_dropout": config.lora_config.get("lora_dropout", 0.05),
        }

        if "target_modules" in config.lora_config:
            peft["lora_target_modules"] = config.lora_config["target_modules"]

        if config.lora_config.get("quantize_base"):
            peft["load_in_4bit"] = True
            peft["bnb_4bit_quant_type"] = "nf4"
            peft["bnb_4bit_compute_dtype"] = _HF_DTYPE_MAP.get(
                config.dtype, "bfloat16"
            )

        return peft

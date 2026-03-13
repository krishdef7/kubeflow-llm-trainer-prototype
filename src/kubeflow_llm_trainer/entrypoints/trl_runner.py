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

"""TRL training entrypoint invoked by the Kubeflow Trainer container.

This script is the ``command`` target in the TRL backend's ``ContainerSpec``.
It reads training configuration from the ``TRL_TRAINING_ARGS`` environment
variable and runs TRL **in-process** so that ``KubeflowTrainerCallback``
can be injected directly into the trainer's callback list for progress
reporting (KEP-2779 / kubeflow/trainer#3227).

Usage (inside the training container)::

    python -m kubeflow_llm_trainer.entrypoints.trl_runner sft
    python -m kubeflow_llm_trainer.entrypoints.trl_runner dpo
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any


def _resolve_paths(training_args: dict[str, Any]) -> None:
    """Override model/dataset paths with Kubeflow PVC mounts if available."""
    model_path = os.environ.get("KUBEFLOW_MODEL_PATH")
    dataset_path = os.environ.get("KUBEFLOW_DATASET_PATH")
    if model_path and os.path.isdir(model_path):
        training_args["model_name_or_path"] = model_path
    if dataset_path and os.path.isdir(dataset_path):
        training_args["dataset_name"] = dataset_path


def _build_callbacks() -> list:
    """Build the callback list, including Kubeflow progress reporting."""
    from kubeflow_llm_trainer.progress import (
        KubeflowTrainerCallback,
        is_progress_reporting_available,
    )
    callbacks = []
    if is_progress_reporting_available():
        callbacks.append(KubeflowTrainerCallback())
        print("Kubeflow progress reporting: enabled")
    else:
        print("Kubeflow progress reporting: disabled (env vars not set)")
    return callbacks


def _load_dataset(dataset_name: str | None):
    """Load a dataset from HuggingFace Hub or local path.

    TRL trainers expect a ``datasets.Dataset`` object, not a string path.
    This function handles the conversion.
    """
    if dataset_name is None:
        return None
    from datasets import load_dataset  # type: ignore[import-untyped]
    # If it's a local directory, load from disk. Otherwise treat as a
    # HuggingFace Hub dataset identifier.
    if os.path.isdir(dataset_name):
        return load_dataset("json", data_dir=dataset_name, split="train")
    return load_dataset(dataset_name, split="train")


def _build_peft_config(training_args: dict[str, Any]):
    """Extract PEFT args from training_args and build a PeftConfig.

    TRL's _build_peft_args() in trl.py generates keys like ``use_peft``,
    ``lora_r``, ``lora_alpha``, etc.  These are NOT fields on TRL's Config
    classes — they need to be extracted and converted into a ``LoraConfig``
    object that gets passed to the trainer constructor separately.

    Returns ``None`` if PEFT is not requested.
    """
    if not training_args.pop("use_peft", False):
        return None

    from peft import LoraConfig  # type: ignore[import-untyped]

    lora_kwargs: dict[str, Any] = {}
    # Map our flat keys to LoraConfig constructor args.
    key_map = {
        "lora_r": "r",
        "lora_alpha": "lora_alpha",
        "lora_dropout": "lora_dropout",
        "lora_target_modules": "target_modules",
    }
    for our_key, peft_key in key_map.items():
        value = training_args.pop(our_key, None)
        if value is not None:
            lora_kwargs[peft_key] = value

    # BitsAndBytes quantization config for QLoRA.
    load_in_4bit = training_args.pop("load_in_4bit", False)
    bnb_quant_type = training_args.pop("bnb_4bit_quant_type", None)
    bnb_compute_dtype = training_args.pop("bnb_4bit_compute_dtype", None)

    return LoraConfig(**lora_kwargs)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m kubeflow_llm_trainer.entrypoints.trl_runner <command>")
        print("Commands: sft, dpo, ppo, orpo, kto")
        sys.exit(1)

    command = sys.argv[1]
    args_json = os.environ.get("TRL_TRAINING_ARGS")
    if not args_json:
        print("ERROR: TRL_TRAINING_ARGS environment variable not set.")
        sys.exit(1)

    training_args = json.loads(args_json)
    _resolve_paths(training_args)
    callbacks = _build_callbacks()

    # Extract fields that are NOT part of TRL Config classes.
    model_name = training_args.pop("model_name_or_path", None)
    dataset_name = training_args.pop("dataset_name", None)
    reward_model_name = training_args.pop("reward_model", None)
    output_dir = training_args.get("output_dir", "/mnt/output")

    # Load the dataset as a datasets.Dataset object.
    # TRL trainers expect Dataset objects, not string paths.
    dataset = _load_dataset(dataset_name)

    # Extract PEFT/LoRA config if present (removes PEFT keys from training_args).
    peft_config = _build_peft_config(training_args)

    # Remaining training_args are all valid TrainingArguments fields.
    # Dispatch to the appropriate TRL trainer.
    if command == "sft":
        from trl import SFTConfig, SFTTrainer  # type: ignore[import-untyped]
        config = SFTConfig(**training_args)
        trainer = SFTTrainer(
            model=model_name,
            args=config,
            train_dataset=dataset,
            peft_config=peft_config,
            callbacks=callbacks,
        )
    elif command == "dpo":
        from trl import DPOConfig, DPOTrainer  # type: ignore[import-untyped]
        config = DPOConfig(**training_args)
        trainer = DPOTrainer(
            model=model_name,
            args=config,
            train_dataset=dataset,
            peft_config=peft_config,
            callbacks=callbacks,
        )
    elif command == "ppo":
        from trl import PPOConfig, PPOTrainer  # type: ignore[import-untyped]
        config = PPOConfig(**training_args)
        trainer = PPOTrainer(
            model=model_name,
            args=config,
            reward_model=reward_model_name,
            train_dataset=dataset,
            callbacks=callbacks,
        )
    elif command == "orpo":
        from trl import ORPOConfig, ORPOTrainer  # type: ignore[import-untyped]
        config = ORPOConfig(**training_args)
        trainer = ORPOTrainer(
            model=model_name,
            args=config,
            train_dataset=dataset,
            peft_config=peft_config,
            callbacks=callbacks,
        )
    elif command == "kto":
        from trl import KTOConfig, KTOTrainer  # type: ignore[import-untyped]
        config = KTOConfig(**training_args)
        trainer = KTOTrainer(
            model=model_name,
            args=config,
            train_dataset=dataset,
            peft_config=peft_config,
            callbacks=callbacks,
        )
    else:
        print(f"ERROR: Unknown TRL command: {command}")
        sys.exit(1)

    print(f"Starting TRL {command} training: model={model_name}")
    trainer.train()
    trainer.save_model(output_dir)
    print("Training complete.")


if __name__ == "__main__":
    main()

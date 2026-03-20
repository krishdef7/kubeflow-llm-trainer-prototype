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


def _load_dataset(dataset_name: str | None, split: str = "train"):
    """Load a dataset from HuggingFace Hub or local path.

    TRL trainers expect a ``datasets.Dataset`` object, not a string path.

    Args:
        dataset_name: HuggingFace Hub ID or local directory path.
        split: Dataset split to load.  Falls back to first available split
            if the requested one doesn't exist.
    """
    if dataset_name is None:
        return None
    from datasets import load_dataset  # type: ignore[import-untyped]

    if os.path.isdir(dataset_name):
        return load_dataset("json", data_dir=dataset_name, split=split)

    try:
        return load_dataset(dataset_name, split=split)
    except (ValueError, KeyError):
        # The requested split doesn't exist (common for DPO preference
        # datasets which use "train_prefs", etc.).  Load the full dataset
        # and use its first split.
        ds = load_dataset(dataset_name)
        first_split = next(iter(ds))
        print(f"Split {split!r} not found, using {first_split!r} instead.")
        return ds[first_split]


def _build_peft_config(training_args: dict[str, Any]) -> tuple[Any, Any]:
    """Extract PEFT/BNB args and build LoraConfig + BitsAndBytesConfig.

    Returns:
        A tuple of ``(peft_config, quantization_config)`` where either may
        be ``None``.  The ``peft_config`` is passed to the TRL trainer's
        ``peft_config=`` parameter.  The ``quantization_config`` is passed
        to ``AutoModelForCausalLM.from_pretrained()`` for QLoRA 4-bit loading.
    """
    if not training_args.pop("use_peft", False):
        return None, None

    from peft import LoraConfig  # type: ignore[import-untyped]

    lora_kwargs: dict[str, Any] = {}
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

    peft_config = LoraConfig(**lora_kwargs)

    # Build BitsAndBytesConfig for QLoRA 4-bit quantization.
    load_in_4bit = training_args.pop("load_in_4bit", False)
    bnb_quant_type = training_args.pop("bnb_4bit_quant_type", None)
    bnb_compute_dtype = training_args.pop("bnb_4bit_compute_dtype", None)

    quantization_config = None
    if load_in_4bit:
        try:
            import torch
            from transformers import BitsAndBytesConfig  # type: ignore[import-untyped]
            bnb_kwargs: dict[str, Any] = {"load_in_4bit": True}
            if bnb_quant_type:
                bnb_kwargs["bnb_4bit_quant_type"] = bnb_quant_type
            if bnb_compute_dtype:
                bnb_kwargs["bnb_4bit_compute_dtype"] = getattr(
                    torch, bnb_compute_dtype, torch.bfloat16
                )
            quantization_config = BitsAndBytesConfig(**bnb_kwargs)
        except ImportError:
            print("WARNING: bitsandbytes not available, skipping 4-bit quantization.")

    return peft_config, quantization_config


def _load_model(model_name: str | None, quantization_config: Any = None):
    """Load a model, optionally with QLoRA quantization.

    When ``quantization_config`` is provided (QLoRA), the model must be
    loaded explicitly with ``from_pretrained`` rather than letting TRL
    handle it via the string model name, because TRL's string-based loading
    doesn't support ``quantization_config``.
    """
    if model_name is None:
        return None
    if quantization_config is None:
        # Let TRL handle model loading from the string name/path.
        return model_name
    # QLoRA: load the model with 4-bit quantization.
    from transformers import AutoModelForCausalLM  # type: ignore[import-untyped]
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
    )


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
    dataset_split = training_args.pop("dataset_split", "train")

    # Load the dataset as a datasets.Dataset object.
    dataset = _load_dataset(dataset_name, split=dataset_split)

    # Extract PEFT/LoRA + BNB quantization configs.
    peft_config, quantization_config = _build_peft_config(training_args)

    # Load model (with QLoRA quantization if requested).
    model = _load_model(model_name, quantization_config)

    # Dispatch to the appropriate TRL trainer.
    if command == "sft":
        from trl import SFTConfig, SFTTrainer  # type: ignore[import-untyped]
        config = SFTConfig(**training_args)
        trainer = SFTTrainer(
            model=model, args=config, train_dataset=dataset,
            peft_config=peft_config, callbacks=callbacks,
        )
    elif command == "dpo":
        from trl import DPOConfig, DPOTrainer  # type: ignore[import-untyped]
        config = DPOConfig(**training_args)
        trainer = DPOTrainer(
            model=model, args=config, train_dataset=dataset,
            peft_config=peft_config, callbacks=callbacks,
        )
    elif command == "ppo":
        from trl import PPOConfig, PPOTrainer  # type: ignore[import-untyped]
        config = PPOConfig(**training_args)
        # PPOTrainer API varies across TRL versions.  In TRL >= 0.12,
        # reward_model is passed to the constructor.
        trainer = PPOTrainer(
            model=model, args=config, train_dataset=dataset,
            reward_model=reward_model_name, callbacks=callbacks,
        )
    elif command == "orpo":
        from trl import ORPOConfig, ORPOTrainer  # type: ignore[import-untyped]
        config = ORPOConfig(**training_args)
        trainer = ORPOTrainer(
            model=model, args=config, train_dataset=dataset,
            peft_config=peft_config, callbacks=callbacks,
        )
    elif command == "kto":
        from trl import KTOConfig, KTOTrainer  # type: ignore[import-untyped]
        config = KTOConfig(**training_args)
        trainer = KTOTrainer(
            model=model, args=config, train_dataset=dataset,
            peft_config=peft_config, callbacks=callbacks,
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

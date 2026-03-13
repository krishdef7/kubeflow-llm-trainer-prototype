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

"""Unsloth training entrypoint invoked by the Kubeflow Trainer container.

This script reads the ``UNSLOTH_TRAINING_CONFIG`` environment variable,
loads the model with ``unsloth.FastLanguageModel``, applies LoRA, and
trains using the appropriate TRL Trainer (SFT, DPO, or ORPO).

Usage (inside the training container)::

    python -m kubeflow_llm_trainer.entrypoints.unsloth_runner
"""

from __future__ import annotations

import json
import os
import sys


def main() -> None:
    config_json = os.environ.get("UNSLOTH_TRAINING_CONFIG")
    trainer_type = os.environ.get("UNSLOTH_TRAINER_TYPE", "sft")

    if not config_json:
        print("ERROR: UNSLOTH_TRAINING_CONFIG environment variable not set.")
        sys.exit(1)

    config = json.loads(config_json)

    # Resolve paths from Kubeflow initializers.
    model_path = os.environ.get("KUBEFLOW_MODEL_PATH")
    dataset_path = os.environ.get("KUBEFLOW_DATASET_PATH")
    if model_path and os.path.isdir(model_path):
        config["model_name"] = model_path
    if dataset_path and os.path.isdir(dataset_path):
        config["dataset_name"] = dataset_path

    # Import Unsloth (available in the container image).
    from unsloth import FastLanguageModel  # type: ignore[import-untyped]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config.get("max_seq_length", 2048),
        dtype=None,  # auto-detect
        load_in_4bit=config.get("load_in_4bit", False),
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 16),
        lora_dropout=config.get("lora_dropout", 0.0),
        target_modules=config.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        use_gradient_checkpointing=config.get(
            "use_gradient_checkpointing", "unsloth"
        ),
        random_state=config.get("random_state", 3407),
        use_rslora=config.get("use_rslora", False),
    )

    # Load dataset.
    from datasets import load_dataset  # type: ignore[import-untyped]
    dataset = load_dataset(config["dataset_name"], split="train")

    # Select trainer based on type.
    from transformers import TrainingArguments  # type: ignore[import-untyped]
    training_args = TrainingArguments(
        output_dir=config.get("output_dir", "/mnt/output"),
        num_train_epochs=config.get("num_train_epochs", 3),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 4),
        learning_rate=config.get("learning_rate", 2e-5),
    )

    # Inject Kubeflow progress reporting callback if the status server
    # env vars are present (TrainJobStatus feature gate enabled).
    # Ref: kubeflow/trainer#3227
    from kubeflow_llm_trainer.progress import (
        KubeflowTrainerCallback,
        is_progress_reporting_available,
    )
    callbacks = []
    if is_progress_reporting_available():
        callbacks.append(KubeflowTrainerCallback())
        print("Kubeflow progress reporting: enabled")

    if trainer_type == "dpo":
        from trl import DPOTrainer  # type: ignore[import-untyped]
        trainer = DPOTrainer(
            model=model, args=training_args, train_dataset=dataset,
            tokenizer=tokenizer, callbacks=callbacks,
        )
    elif trainer_type == "orpo":
        from trl import ORPOTrainer  # type: ignore[import-untyped]
        trainer = ORPOTrainer(
            model=model, args=training_args, train_dataset=dataset,
            tokenizer=tokenizer, callbacks=callbacks,
        )
    else:
        from trl import SFTTrainer  # type: ignore[import-untyped]
        trainer = SFTTrainer(
            model=model, args=training_args, train_dataset=dataset,
            tokenizer=tokenizer, callbacks=callbacks,
        )

    trainer.train()
    model.save_pretrained(config.get("output_dir", "/mnt/output"))
    print("Training complete.")


if __name__ == "__main__":
    main()

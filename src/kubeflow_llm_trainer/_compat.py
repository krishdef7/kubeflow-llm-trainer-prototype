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

"""Backward compatibility: convert BuiltinTrainer + TorchTuneConfig → LLMTrainer.

This module provides a bridge so that existing code using ``BuiltinTrainer``
continues to work after the Dynamic LLM Trainer Framework is adopted.

The migration is transparent — see the ``adapt_builtin_trainer()`` docstring
for the integration plan.

.. warning::

    ``TorchTuneConfig.model_args`` contains a TorchTune *recipe identifier*
    (e.g. ``"llama3_2/1B_lora_single_device"``), NOT a HuggingFace model ID.
    The adapter passes this value through as ``LLMConfig.model``, which means
    the TorchTune backend will use it as the ``--config`` base config name.
    Users migrating to ``LLMConfig`` directly should use the actual HuggingFace
    model ID (e.g. ``"meta-llama/Llama-3.2-1B-Instruct"``) and set the
    TorchTune config name via ``extra_args["torchtune_config"]``.
"""

from __future__ import annotations

import logging
from typing import Any

from kubeflow_llm_trainer.interface import (
    DataType,
    FineTuningMethod,
    LLMConfig,
)
from kubeflow_llm_trainer.trainer import LLMTrainer

logger = logging.getLogger(__name__)


def adapt_builtin_trainer(
    builtin: Any,
    *,
    model: str | None = None,
    dataset: str | None = None,
) -> LLMTrainer:
    """Convert a ``BuiltinTrainer`` into an ``LLMTrainer``.

    This adapter extracts parameters from the legacy ``TorchTuneConfig`` and
    maps them to the backend-agnostic ``LLMConfig``.

    Integration plan: in ``TrainerClient.train()``, the Kubernetes backend
    transparently converts before dispatch::

        if isinstance(trainer, BuiltinTrainer):
            trainer = adapt_builtin_trainer(trainer)
        if isinstance(trainer, LLMTrainer):
            resolved = trainer.resolve()
            ...

    .. note::

        ``TorchTuneConfig.model_args`` is a TorchTune recipe identifier
        (e.g. ``"llama3_2/1B_lora_single_device"``), not a HuggingFace model
        ID.  This adapter passes it through as-is.  When migrating to
        ``LLMConfig`` directly, users should provide the HuggingFace model ID
        and set ``extra_args["torchtune_config"]`` for the base config.

    Args:
        builtin: The existing ``BuiltinTrainer`` instance.
        model: Optional model ID override.
        dataset: Optional dataset ID override.

    Returns:
        An ``LLMTrainer`` configured for the ``torchtune`` backend.

    Raises:
        TypeError: If the config is not a ``TorchTuneConfig``-like object.
    """
    config = builtin.config

    if not hasattr(config, "model_args"):
        raise TypeError(
            f"Expected a TorchTuneConfig-like object, got {type(config)!r}. "
            f"adapt_builtin_trainer() only works with TorchTuneConfig."
        )

    # Extract model and dataset.
    # NOTE: model_args is a TorchTune recipe identifier, not a HuggingFace
    # model ID. This is a lossy mapping — see module docstring.
    resolved_model = model or config.model_args or "unknown-model"
    resolved_dataset = dataset or config.dataset_args or "unknown-dataset"

    if not model and config.model_args:
        logger.warning(
            "adapt_builtin_trainer: using TorchTuneConfig.model_args=%r as "
            "LLMConfig.model. This is a TorchTune recipe identifier, not a "
            "HuggingFace model ID — the mapping is lossy.",
            config.model_args,
        )

    # Determine fine-tuning method from LoRA config presence.
    method = FineTuningMethod.SFT
    lora_config: dict[str, Any] | None = None

    if hasattr(config, "peft_config") and config.peft_config is not None:
        peft = config.peft_config
        method = FineTuningMethod.LORA
        lora_config = {}
        if hasattr(peft, "lora_rank"):
            lora_config["r"] = peft.lora_rank
        if hasattr(peft, "lora_alpha"):
            lora_config["lora_alpha"] = peft.lora_alpha
        if hasattr(peft, "lora_dropout"):
            lora_config["lora_dropout"] = peft.lora_dropout
        if hasattr(peft, "lora_attn_modules"):
            lora_config["lora_attn_modules"] = peft.lora_attn_modules
        if hasattr(peft, "apply_lora_to_mlp"):
            lora_config["apply_lora_to_mlp"] = peft.apply_lora_to_mlp
        if hasattr(peft, "quantize_base") and peft.quantize_base:
            method = FineTuningMethod.QLORA
            lora_config["quantize_base"] = True

    # Map dtype.
    dtype = DataType.BF16
    if hasattr(config, "dtype") and config.dtype is not None:
        dtype_str = str(config.dtype).lower()
        if "fp16" in dtype_str or "float16" in dtype_str:
            dtype = DataType.FP16
        elif "fp32" in dtype_str or "float32" in dtype_str:
            dtype = DataType.FP32

    llm_config = LLMConfig(
        model=resolved_model,
        dataset=resolved_dataset,
        method=method,
        backend_name="torchtune",
        num_nodes=config.num_nodes or 1,
        dtype=dtype,
        epochs=config.epochs or 3,
        batch_size=config.batch_size or 4,
        lora_config=lora_config,
    )

    return LLMTrainer(
        config=llm_config,
        resources_per_node=config.resources_per_node or {},
    )

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

"""Abstract base class for pluggable LLM fine-tuning backends.

This module defines the ``LLMBackend`` protocol that every backend — both
in-tree (TorchTune, TRL, Unsloth) and external — must implement. The interface
is intentionally minimal: a backend must be able to produce a container spec
(image, command, args, env) and declare its required packages. Everything else
is handled by the SDK-layer ``LLMTrainer`` and the Kubeflow Trainer controller.

Design principles
-----------------
1. **Symmetry with TrainingRuntime on the control plane.** Just as
   ``ClusterTrainingRuntime`` defines runtime blueprints on the Kubernetes side,
   ``LLMBackend`` defines fine-tuning strategies on the SDK side.

2. **Config-driven, not function-driven.** Unlike ``CustomTrainer`` (which wraps
   a user-provided function), LLM backends are config objects: the user
   specifies *what* to train (model, dataset, method, hyperparameters) and the
   backend decides *how* (entrypoint, container image, framework args).

3. **Zero controller changes for new backends.** A new backend only needs to
   produce a valid ``ContainerSpec``; the existing torch plugin in
   ``pkg/runtime/framework/plugins/torch/`` handles the rest.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums shared across backends
# ---------------------------------------------------------------------------

class FineTuningMethod(str, Enum):
    """Supported fine-tuning strategies."""

    SFT = "sft"
    DPO = "dpo"
    PPO = "ppo"
    ORPO = "orpo"
    KTO = "kto"
    FULL = "full"
    LORA = "lora"
    QLORA = "qlora"


class DataType(str, Enum):
    """Torch dtypes exposed to users."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


# ---------------------------------------------------------------------------
# Container spec — what the SDK sends to the controller
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContainerSpec:
    """Immutable description of the training container.

    This is the contract between the SDK and the Kubeflow Trainer controller.
    The controller uses these fields to build the Pod template for the trainer
    step of the TrainJob.

    Attributes:
        image: Container image URI (e.g. ``ghcr.io/kubeflow/trl-trainer:latest``).
        command: Entrypoint command (e.g. ``["python", "-m", "trl"]``).
        args: CLI arguments appended to the command.
        env: Environment variables as ``{name: value}`` pairs.
    """

    image: str
    command: list[str]
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# LLMBackend abstract base class
# ---------------------------------------------------------------------------

class LLMBackend(abc.ABC):
    """Protocol for pluggable LLM fine-tuning backends.

    Every backend must implement three methods:

    * ``name``        — unique identifier used in the registry.
    * ``validate``    — raise ``ValueError`` if the config is invalid.
    * ``to_container_spec`` — produce a ``ContainerSpec`` for the TrainJob.

    And may optionally override:

    * ``packages_to_install`` — extra pip packages for the runtime image.
    * ``supported_methods``   — which ``FineTuningMethod`` values are supported.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique backend identifier (e.g. ``"torchtune"``, ``"trl"``)."""

    @abc.abstractmethod
    def validate(self, config: "LLMConfig") -> None:
        """Validate the user-provided config.

        Raise ``ValueError`` with a descriptive message if the config is
        incompatible with this backend.
        """

    @abc.abstractmethod
    def to_container_spec(self, config: "LLMConfig") -> ContainerSpec:
        """Convert an ``LLMConfig`` into a ``ContainerSpec``.

        This is the core translation step: the backend maps user-facing
        parameters (model, dataset, method, hyperparams) into the concrete
        container image, entrypoint, and CLI args that the Kubeflow Trainer
        controller will use to build the trainer Pod.
        """

    @property
    def packages_to_install(self) -> list[str]:
        """Extra pip packages required at training time."""
        return []

    @property
    def supported_methods(self) -> list[FineTuningMethod]:
        """Fine-tuning methods this backend supports.

        Returning an empty list means *all* methods are allowed (no filtering).
        """
        return []

    @property
    def framework(self) -> str:
        """The ``trainer.kubeflow.org/framework`` label value.

        Defaults to ``"torch"`` because most LLM backends are PyTorch-based.
        """
        return "torch"

    def __repr__(self) -> str:
        methods = ", ".join(m.value for m in self.supported_methods) or "all"
        return f"<{type(self).__name__}(name={self.name!r}, methods=[{methods}])>"


# ---------------------------------------------------------------------------
# LLMConfig — the user-facing configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    """Backend-agnostic configuration for LLM fine-tuning.

    This dataclass is the single entry point for users.  They set the model,
    dataset, fine-tuning method, and hyperparameters; the ``LLMTrainer``
    dispatches to the correct ``LLMBackend`` to convert it into a
    ``ContainerSpec``.

    Attributes:
        model: HuggingFace model ID or local path.
        dataset: HuggingFace dataset ID or local path.
        method: Fine-tuning strategy. Defaults to SFT.
        backend_name: Which backend to use. ``None`` → registry default.
        num_nodes: Number of training nodes.  Affects recipe selection
            (e.g. single-device vs distributed in TorchTune).
        dtype: Training precision.
        epochs: Number of training epochs.
        batch_size: Per-device batch size.
        learning_rate: Optimizer learning rate.
        lora_config: LoRA / QLoRA configuration dict.  Backend-specific keys.
        extra_args: Escape hatch for backend-specific knobs that don't fit
            in the common schema.
    """

    model: str
    dataset: str
    method: FineTuningMethod = FineTuningMethod.SFT
    backend_name: str | None = None
    num_nodes: int = 1
    dtype: DataType = DataType.BF16
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-5
    lora_config: dict[str, Any] | None = None
    extra_args: dict[str, Any] = field(default_factory=dict)

    @property
    def is_peft(self) -> bool:
        """Whether this config uses parameter-efficient fine-tuning."""
        return self.method in (
            FineTuningMethod.LORA,
            FineTuningMethod.QLORA,
        ) or self.lora_config is not None

    @property
    def is_distributed(self) -> bool:
        """Whether this config targets multi-node training."""
        return self.num_nodes > 1

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("LLMConfig.model must not be empty.")
        if not self.dataset:
            raise ValueError("LLMConfig.dataset must not be empty.")
        if self.num_nodes < 1:
            raise ValueError(
                f"LLMConfig.num_nodes must be >= 1, got {self.num_nodes}."
            )
        if self.epochs < 1:
            raise ValueError(f"LLMConfig.epochs must be >= 1, got {self.epochs}.")
        if self.batch_size < 1:
            raise ValueError(
                f"LLMConfig.batch_size must be >= 1, got {self.batch_size}."
            )
        if self.learning_rate <= 0:
            raise ValueError(
                f"LLMConfig.learning_rate must be > 0, got {self.learning_rate}."
            )

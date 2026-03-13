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

"""Dynamic LLM Trainer Framework for Kubeflow.

GSoC 2026 Prototype — KEP-2839: Kubeflow Dynamic LLM Trainer Framework.

This package provides a pluggable backend architecture for LLM fine-tuning
in the Kubeflow SDK, decoupling the SDK from any single framework.

Quick start::

    from kubeflow_llm_trainer import (
        LLMTrainer, LLMConfig, FineTuningMethod, BackendRegistry,
    )

    # Register in-tree backends (done automatically via entry points in
    # the real SDK; explicit here for prototype clarity).
    from kubeflow_llm_trainer.backends.torchtune import TorchTuneBackend
    from kubeflow_llm_trainer.backends.trl import TRLBackend
    from kubeflow_llm_trainer.backends.unsloth import UnslothBackend
    BackendRegistry.register(TorchTuneBackend())
    BackendRegistry.register(TRLBackend())
    BackendRegistry.register(UnslothBackend())

    # Create a trainer.
    trainer = LLMTrainer(
        config=LLMConfig(
            model="meta-llama/Llama-3.2-1B-Instruct",
            dataset="tatsu-lab/alpaca",
            method=FineTuningMethod.SFT,
            backend_name="trl",
            lora_config={"r": 16, "lora_alpha": 32},
        ),
        resources_per_node={"gpu": 2},
    )

    # Resolve to get the container spec.
    resolved = trainer.resolve()
    print(resolved.container_spec.image)
    print(resolved.container_spec.command)
"""

from kubeflow_llm_trainer.interface import (
    ContainerSpec,
    DataType,
    FineTuningMethod,
    LLMBackend,
    LLMConfig,
)
from kubeflow_llm_trainer.registry import BackendRegistry
from kubeflow_llm_trainer.trainer import LLMTrainer, ResolvedLLMTrainer
from kubeflow_llm_trainer.progress import (
    KubeflowProgressReporter,
    KubeflowTrainerCallback,
    Metric,
    TrainerStatus,
    is_progress_reporting_available,
)

__all__ = [
    # Core types
    "LLMTrainer",
    "LLMConfig",
    "LLMBackend",
    "ContainerSpec",
    "ResolvedLLMTrainer",
    # Enums
    "FineTuningMethod",
    "DataType",
    # Registry
    "BackendRegistry",
    # Progress reporting (KEP-2779 / kubeflow/trainer#3227)
    "KubeflowProgressReporter",
    "KubeflowTrainerCallback",
    "TrainerStatus",
    "Metric",
    "is_progress_reporting_available",
]

__version__ = "0.1.0"

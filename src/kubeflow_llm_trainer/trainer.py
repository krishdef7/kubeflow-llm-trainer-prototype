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

"""LLMTrainer — the SDK-layer replacement for BuiltinTrainer's hardcoded dispatch.

Today, ``BuiltinTrainer`` contains an ``isinstance(trainer.config, TorchTuneConfig)``
check that couples the SDK to a single fine-tuning framework.  ``LLMTrainer`` removes
that coupling: it accepts any ``LLMConfig``, resolves the appropriate ``LLMBackend``
from the registry, validates the config against it, and produces a ``ContainerSpec``
that the ``TrainerClient`` can embed into a ``TrainJob`` spec.

Integration with ``TrainerClient.train()``
------------------------------------------

The existing ``train()`` method accepts ``trainer`` as one of::

    Union[CustomTrainer, BuiltinTrainer, BaseTrainer, None]

This prototype proposes extending that union to include ``LLMTrainer``::

    Union[CustomTrainer, BuiltinTrainer, BaseTrainer, LLMTrainer, None]

Internally, the Kubernetes backend's ``_build_trainjob_spec`` method checks
``isinstance(trainer, LLMTrainer)`` and calls ``trainer.resolve()`` to obtain
the ``ContainerSpec``.  Backward compatibility with ``BuiltinTrainer`` is
preserved: see ``_compat.py``.

Relationship to KEP-285 (Specialized Trainers)
-----------------------------------------------

KEP-285's ``BaseTrainer`` is *function-driven* — the user provides a training
function and the framework (Torch, MPI, JAX) handles distribution.

``LLMTrainer`` is *config-driven* — the user provides parameters and the
backend handles the training logic.

These are complementary:

* ``TorchTrainer``  (KEP-285)  → "run *my code* on N nodes"
* ``LLMTrainer``    (KEP-2839) → "fine-tune *this model* with *these params*"

The two can coexist in the same SDK; ``LLMTrainer`` conceptually sits on top
of a ``TorchTrainer`` runtime since most LLM backends use PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kubeflow_llm_trainer.interface import ContainerSpec, LLMConfig
from kubeflow_llm_trainer.registry import BackendRegistry


@dataclass
class LLMTrainer:
    """Config-driven LLM trainer for the Kubeflow SDK.

    This is the user-facing type passed to ``TrainerClient.train(trainer=...)``.

    Args:
        config: The fine-tuning configuration (includes num_nodes).
        resources_per_node: Resource requests per training node
            (e.g. ``{"gpu": 2, "memory": "64Gi"}``).
        packages_to_install: Additional pip packages to install at runtime.
            Backend-required packages are injected automatically.

    Example::

        from kubeflow.trainer import TrainerClient
        from kubeflow_llm_trainer import LLMTrainer, LLMConfig, FineTuningMethod

        client = TrainerClient()
        client.train(
            name="llama-dpo",
            trainer=LLMTrainer(
                config=LLMConfig(
                    model="meta-llama/Llama-3.2-1B-Instruct",
                    dataset="trl-lib/ultrafeedback_binarized",
                    method=FineTuningMethod.DPO,
                    backend_name="trl",
                    num_nodes=2,
                    lora_config={"r": 16, "lora_alpha": 32},
                ),
                resources_per_node={"gpu": 2},
            ),
        )
    """

    config: LLMConfig
    resources_per_node: dict[str, Any] = field(default_factory=dict)
    packages_to_install: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # num_nodes validation is handled by LLMConfig.__post_init__.
        pass

    def resolve(self) -> ResolvedLLMTrainer:
        """Resolve the config against the backend registry.

        This is called by the ``TrainerClient`` Kubernetes backend when
        building the ``TrainJob`` spec.

        Returns:
            A ``ResolvedLLMTrainer`` containing the validated ``ContainerSpec``
            and merged package list.

        Raises:
            KeyError: If the requested backend is not registered.
            ValueError: If the config fails backend validation.
        """
        backend = (
            BackendRegistry.get(self.config.backend_name)
            if self.config.backend_name
            else BackendRegistry.get_default()
        )

        # Validate the config against the backend.
        backend.validate(self.config)

        # Check method compatibility.
        if backend.supported_methods and self.config.method not in backend.supported_methods:
            supported = ", ".join(m.value for m in backend.supported_methods)
            raise ValueError(
                f"Backend {backend.name!r} does not support method "
                f"{self.config.method.value!r}. Supported methods: {supported}."
            )

        # Produce the container spec.
        container_spec = backend.to_container_spec(self.config)

        # Merge package lists: backend defaults + user overrides.
        all_packages = list(dict.fromkeys(
            backend.packages_to_install + self.packages_to_install
        ))

        return ResolvedLLMTrainer(
            container_spec=container_spec,
            backend_name=backend.name,
            framework=backend.framework,
            packages_to_install=all_packages,
            resources_per_node=self.resources_per_node,
            num_nodes=self.config.num_nodes,
        )


@dataclass(frozen=True)
class ResolvedLLMTrainer:
    """Output of ``LLMTrainer.resolve()``.

    This is an internal type consumed by the ``TrainerClient`` Kubernetes
    backend.  It is not exposed to users.

    The ``TrainerClient`` uses these fields to populate:

    * ``TrainJob.spec.trainer.image``     ← ``container_spec.image``
    * ``TrainJob.spec.trainer.command``    ← ``container_spec.command``
    * ``TrainJob.spec.trainer.args``       ← ``container_spec.args``
    * ``TrainJob.spec.trainer.env``        ← ``container_spec.env``
    * ``TrainJob.spec.runtimeRef``         ← looked up via ``framework`` label
    * Pip init container                   ← ``packages_to_install``
    """

    container_spec: ContainerSpec
    backend_name: str
    framework: str
    packages_to_install: list[str]
    resources_per_node: dict[str, Any]
    num_nodes: int

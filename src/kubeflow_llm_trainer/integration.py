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

"""Integration with the Kubeflow SDK ``TrainerClient``.

This module shows the *exact changes* required in the ``TrainerClient`` and
its Kubernetes backend to support the Dynamic LLM Trainer Framework.

The changes are minimal — approximately **40 lines of diff** in the existing
codebase.

Changes to ``kubeflow/sdk/kubeflow/trainer/trainer_client.py``
---------------------------------------------------------------

1. Add ``LLMTrainer`` to the ``trainer`` parameter type union.
2. In the Kubernetes backend's ``_build_trainjob_spec``, add a branch for
   ``isinstance(trainer, LLMTrainer)``.

Changes to ``kubeflow/trainer/pkg/runtime/framework/plugins/torch/torch.go``
------------------------------------------------------------------------------

None.  The torch plugin already handles arbitrary container specs.
The ``LLMTrainer.resolve()`` output is consumed entirely on the SDK side;
the controller sees a normal ``TrainJob`` spec with container image/command/args.

Example diff
------------

This is the *conceptual* diff against the current SDK code:

.. code-block:: diff

    # kubeflow/sdk/kubeflow/trainer/trainer_client.py

    + from kubeflow_llm_trainer import LLMTrainer
    + from kubeflow_llm_trainer._compat import adapt_builtin_trainer

      def train(
          self,
          name: Optional[str] = None,
          runtime: Optional[Union[str, "Runtime"]] = None,
          trainer: Optional[Union[
              "CustomTrainer",
              "BuiltinTrainer",
    +         "LLMTrainer",
          ]] = None,
          ...
      ):
    +     # Transparently upgrade BuiltinTrainer to LLMTrainer.
    +     if isinstance(trainer, BuiltinTrainer):
    +         trainer = adapt_builtin_trainer(trainer)
    +
    +     if isinstance(trainer, LLMTrainer):
    +         resolved = trainer.resolve()
    +         # Use resolved.container_spec to populate the TrainJob spec.
    +         return self._submit_llm_trainjob(name, runtime, resolved, ...)
    +
          ...  # existing CustomTrainer path unchanged
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kubeflow_llm_trainer.trainer import LLMTrainer, ResolvedLLMTrainer


@dataclass
class TrainJobSpec:
    """Simplified representation of the Kubernetes TrainJob spec.

    This is a *prototype-only* type that mirrors the structure of the actual
    ``TrainJob`` Kubernetes resource.  In the real SDK, this maps to the
    auto-generated ``kubeflow_trainer_api`` models.

    Shown here to demonstrate the mapping from ``ResolvedLLMTrainer`` fields
    to ``TrainJob`` spec fields.
    """

    name: str
    runtime_ref: str
    trainer_image: str
    trainer_command: list[str]
    trainer_args: list[str]
    trainer_env: dict[str, str]
    num_nodes: int
    resources_per_node: dict[str, Any]
    packages_to_install: list[str]
    labels: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict matching the TrainJob YAML structure."""
        spec: dict[str, Any] = {
            "apiVersion": "trainer.kubeflow.org/v1",
            "kind": "TrainJob",
            "metadata": {
                "name": self.name,
                "labels": self.labels,
            },
            "spec": {
                "runtimeRef": {
                    "name": self.runtime_ref,
                },
                "trainer": {
                    "image": self.trainer_image,
                    "command": self.trainer_command,
                    "args": self.trainer_args,
                    "env": [
                        {"name": k, "value": v}
                        for k, v in self.trainer_env.items()
                    ],
                    "numNodes": self.num_nodes,
                    "resourcesPerNode": self.resources_per_node,
                },
                "managedBy": "kubeflow.org/llm-trainer",
            },
        }
        # Inject a pip init container for backend-required packages.
        # In the real SDK, this maps to the TrainJob's packages_to_install
        # field which the controller converts to an init container.
        if self.packages_to_install:
            spec["spec"]["trainer"]["packagesToInstall"] = self.packages_to_install
        return spec


def build_trainjob_spec(
    name: str,
    resolved: ResolvedLLMTrainer,
    runtime_name: str | None = None,
) -> TrainJobSpec:
    """Build a ``TrainJobSpec`` from a ``ResolvedLLMTrainer``.

    This function demonstrates the exact mapping that would happen inside
    the Kubernetes backend of ``TrainerClient``.

    Args:
        name: TrainJob name.
        resolved: Output of ``LLMTrainer.resolve()``.
        runtime_name: Explicit runtime name override.  If ``None``, the
            runtime is auto-discovered via the framework label.

    Returns:
        A ``TrainJobSpec`` ready for submission.
    """
    # If no runtime name is given, the SDK would list runtimes filtered by
    # the label trainer.kubeflow.org/framework=<resolved.framework> and pick
    # the first one.  This mirrors the auto-discovery logic in KEP-285.
    effective_runtime = runtime_name or f"auto:{resolved.framework}"

    return TrainJobSpec(
        name=name,
        runtime_ref=effective_runtime,
        trainer_image=resolved.container_spec.image,
        trainer_command=resolved.container_spec.command,
        trainer_args=resolved.container_spec.args,
        trainer_env=resolved.container_spec.env,
        num_nodes=resolved.num_nodes,
        resources_per_node=resolved.resources_per_node,
        packages_to_install=resolved.packages_to_install,
        labels={
            "trainer.kubeflow.org/llm-backend": resolved.backend_name,
        },
    )

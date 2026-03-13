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

"""In-tree LLM backend implementations.

Each module in this package provides one ``LLMBackend`` subclass:

* ``torchtune`` — wraps the existing TorchTune training runtime (backward compat).
* ``trl``       — Hugging Face TRL for SFT, DPO, PPO, ORPO, KTO.
* ``unsloth``   — Unsloth for ~2× faster, ~70% lower memory fine-tuning.
"""

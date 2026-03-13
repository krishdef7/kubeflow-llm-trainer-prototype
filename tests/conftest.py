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

"""Shared test fixtures for the Dynamic LLM Trainer Framework."""

from __future__ import annotations

import pytest

from kubeflow_llm_trainer import BackendRegistry, LLMConfig, FineTuningMethod
from kubeflow_llm_trainer.backends.torchtune import TorchTuneBackend
from kubeflow_llm_trainer.backends.trl import TRLBackend
from kubeflow_llm_trainer.backends.unsloth import UnslothBackend


@pytest.fixture(autouse=True)
def clean_registry():
    """Reset the backend registry before each test."""
    BackendRegistry._reset()
    yield
    BackendRegistry._reset()


@pytest.fixture()
def register_all_backends():
    """Register all in-tree backends."""
    BackendRegistry.register(TorchTuneBackend())
    BackendRegistry.register(TRLBackend())
    BackendRegistry.register(UnslothBackend())


@pytest.fixture()
def sft_config() -> LLMConfig:
    """Standard SFT config for testing."""
    return LLMConfig(
        model="meta-llama/Llama-3.2-1B-Instruct",
        dataset="tatsu-lab/alpaca",
        method=FineTuningMethod.SFT,
        epochs=3,
        batch_size=4,
    )


@pytest.fixture()
def dpo_config() -> LLMConfig:
    """Standard DPO config for testing."""
    return LLMConfig(
        model="meta-llama/Llama-3.2-1B-Instruct",
        dataset="trl-lib/ultrafeedback_binarized",
        method=FineTuningMethod.DPO,
        lora_config={"r": 16, "lora_alpha": 32},
    )


@pytest.fixture()
def lora_config() -> LLMConfig:
    """Standard LoRA config for testing."""
    return LLMConfig(
        model="meta-llama/Llama-3.2-1B-Instruct",
        dataset="tatsu-lab/alpaca",
        method=FineTuningMethod.LORA,
        lora_config={
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "lora_attn_modules": ["q_proj", "k_proj", "v_proj"],
        },
    )

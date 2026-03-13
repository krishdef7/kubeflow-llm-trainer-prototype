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

"""Tests for backward compatibility with existing BuiltinTrainer API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from kubeflow_llm_trainer import BackendRegistry, FineTuningMethod
from kubeflow_llm_trainer._compat import adapt_builtin_trainer
from kubeflow_llm_trainer.backends.torchtune import TorchTuneBackend


# ---------------------------------------------------------------------------
# Mock types that mirror the existing SDK types
# ---------------------------------------------------------------------------

@dataclass
class MockLoraConfig:
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_attn_modules: list[str] | None = None
    apply_lora_to_mlp: bool = False
    quantize_base: bool = False


@dataclass
class MockTorchTuneConfig:
    model_args: str | None = None
    dataset_args: str | None = None
    dtype: str | None = "bf16"
    batch_size: int | None = 4
    epochs: int | None = 3
    loss: str | None = None
    peft_config: MockLoraConfig | None = None
    dataset_preprocess_config: Any = None
    resources_per_node: dict[str, Any] | None = None
    num_nodes: int | None = 1


@dataclass
class MockBuiltinTrainer:
    config: MockTorchTuneConfig


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAdaptBuiltinTrainer:
    """Test the BuiltinTrainer → LLMTrainer adapter."""

    @pytest.fixture(autouse=True)
    def _register(self):
        BackendRegistry.register(TorchTuneBackend())

    def test_basic_sft_adaptation(self):
        builtin = MockBuiltinTrainer(
            config=MockTorchTuneConfig(
                model_args="llama3_2/1B",
                dataset_args="alpaca_dataset",
                epochs=5,
                batch_size=8,
            )
        )
        llm_trainer = adapt_builtin_trainer(builtin)
        assert llm_trainer.config.model == "llama3_2/1B"
        assert llm_trainer.config.dataset == "alpaca_dataset"
        assert llm_trainer.config.epochs == 5
        assert llm_trainer.config.batch_size == 8
        assert llm_trainer.config.backend_name == "torchtune"
        assert llm_trainer.config.method == FineTuningMethod.SFT

    def test_lora_adaptation(self):
        builtin = MockBuiltinTrainer(
            config=MockTorchTuneConfig(
                model_args="llama3_2/1B",
                dataset_args="alpaca_dataset",
                peft_config=MockLoraConfig(
                    lora_rank=16,
                    lora_alpha=32,
                    lora_dropout=0.05,
                ),
            )
        )
        llm_trainer = adapt_builtin_trainer(builtin)
        assert llm_trainer.config.method == FineTuningMethod.LORA
        assert llm_trainer.config.lora_config is not None
        assert llm_trainer.config.lora_config["r"] == 16
        assert llm_trainer.config.lora_config["lora_alpha"] == 32
        assert llm_trainer.config.lora_config["lora_dropout"] == 0.05

    def test_qlora_adaptation(self):
        builtin = MockBuiltinTrainer(
            config=MockTorchTuneConfig(
                model_args="llama3_2/1B",
                dataset_args="alpaca_dataset",
                peft_config=MockLoraConfig(
                    lora_rank=4,
                    quantize_base=True,
                ),
            )
        )
        llm_trainer = adapt_builtin_trainer(builtin)
        assert llm_trainer.config.method == FineTuningMethod.QLORA
        assert llm_trainer.config.lora_config["quantize_base"] is True

    def test_dtype_mapping_fp16(self):
        builtin = MockBuiltinTrainer(
            config=MockTorchTuneConfig(
                model_args="test",
                dataset_args="test",
                dtype="fp16",
            )
        )
        from kubeflow_llm_trainer import DataType
        llm_trainer = adapt_builtin_trainer(builtin)
        assert llm_trainer.config.dtype == DataType.FP16

    def test_dtype_mapping_fp32(self):
        builtin = MockBuiltinTrainer(
            config=MockTorchTuneConfig(
                model_args="test",
                dataset_args="test",
                dtype="fp32",
            )
        )
        from kubeflow_llm_trainer import DataType
        llm_trainer = adapt_builtin_trainer(builtin)
        assert llm_trainer.config.dtype == DataType.FP32

    def test_model_override(self):
        builtin = MockBuiltinTrainer(
            config=MockTorchTuneConfig(
                model_args="old-model",
                dataset_args="test",
            )
        )
        llm_trainer = adapt_builtin_trainer(
            builtin, model="meta-llama/Llama-3.2-1B-Instruct"
        )
        assert llm_trainer.config.model == "meta-llama/Llama-3.2-1B-Instruct"

    def test_resources_propagation(self):
        builtin = MockBuiltinTrainer(
            config=MockTorchTuneConfig(
                model_args="test",
                dataset_args="test",
                resources_per_node={"gpu": 4},
                num_nodes=2,
            )
        )
        llm_trainer = adapt_builtin_trainer(builtin)
        assert llm_trainer.resources_per_node == {"gpu": 4}
        assert llm_trainer.config.num_nodes == 2

    def test_adapted_trainer_resolves_successfully(self):
        builtin = MockBuiltinTrainer(
            config=MockTorchTuneConfig(
                model_args="llama3_2/1B",
                dataset_args="alpaca_dataset",
            )
        )
        llm_trainer = adapt_builtin_trainer(builtin)
        resolved = llm_trainer.resolve()
        assert resolved.backend_name == "torchtune"
        assert resolved.container_spec.command == ["tune", "run"]

    def test_rejects_non_torchtune_config(self):
        class BadConfig:
            pass

        class BadTrainer:
            config = BadConfig()

        with pytest.raises(TypeError, match="TorchTuneConfig-like"):
            adapt_builtin_trainer(BadTrainer())  # type: ignore

    def test_defaults_for_missing_fields(self):
        builtin = MockBuiltinTrainer(
            config=MockTorchTuneConfig(
                model_args=None,
                dataset_args=None,
                epochs=None,
                batch_size=None,
                num_nodes=None,
            )
        )
        llm_trainer = adapt_builtin_trainer(builtin)
        assert llm_trainer.config.model == "unknown-model"
        assert llm_trainer.config.dataset == "unknown-dataset"
        assert llm_trainer.config.epochs == 3
        assert llm_trainer.config.batch_size == 4
        assert llm_trainer.config.num_nodes == 1

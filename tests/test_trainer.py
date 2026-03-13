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

"""Tests for LLMTrainer resolution and TrainJob spec generation."""

from __future__ import annotations

import pytest

from kubeflow_llm_trainer import (
    BackendRegistry,
    FineTuningMethod,
    LLMConfig,
    LLMTrainer,
)
from kubeflow_llm_trainer.backends.torchtune import TorchTuneBackend
from kubeflow_llm_trainer.backends.trl import TRLBackend
from kubeflow_llm_trainer.backends.unsloth import UnslothBackend
from kubeflow_llm_trainer.integration import build_trainjob_spec


# =========================================================================
# LLMConfig validation
# =========================================================================


class TestLLMConfig:
    """Tests for LLMConfig validation."""

    def test_valid_config(self, sft_config):
        assert sft_config.model == "meta-llama/Llama-3.2-1B-Instruct"

    def test_empty_model_raises(self):
        with pytest.raises(ValueError, match="model must not be empty"):
            LLMConfig(model="", dataset="test")

    def test_empty_dataset_raises(self):
        with pytest.raises(ValueError, match="dataset must not be empty"):
            LLMConfig(model="test", dataset="")

    def test_invalid_epochs(self):
        with pytest.raises(ValueError, match="epochs must be >= 1"):
            LLMConfig(model="test", dataset="test", epochs=0)

    def test_invalid_batch_size(self):
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            LLMConfig(model="test", dataset="test", batch_size=0)

    def test_invalid_learning_rate(self):
        with pytest.raises(ValueError, match="learning_rate must be > 0"):
            LLMConfig(model="test", dataset="test", learning_rate=-1e-5)

    def test_is_peft_with_lora_config(self):
        config = LLMConfig(
            model="test", dataset="test",
            lora_config={"r": 8},
        )
        assert config.is_peft is True

    def test_is_peft_with_lora_method(self):
        config = LLMConfig(
            model="test", dataset="test",
            method=FineTuningMethod.LORA,
        )
        assert config.is_peft is True

    def test_is_not_peft_sft(self, sft_config):
        assert sft_config.is_peft is False

    def test_is_distributed_false_by_default(self, sft_config):
        assert sft_config.is_distributed is False

    def test_is_distributed_true_for_multi_node(self):
        config = LLMConfig(model="test", dataset="test", num_nodes=2)
        assert config.is_distributed is True

    def test_invalid_num_nodes(self):
        with pytest.raises(ValueError, match="num_nodes must be >= 1"):
            LLMConfig(model="test", dataset="test", num_nodes=0)


# =========================================================================
# LLMTrainer.resolve()
# =========================================================================


class TestLLMTrainerResolve:
    """Tests for the resolve() method — the core dispatch logic."""

    def test_resolve_with_explicit_backend(self, sft_config):
        BackendRegistry.register(TorchTuneBackend())
        sft_config.backend_name = "torchtune"
        trainer = LLMTrainer(config=sft_config)
        resolved = trainer.resolve()
        assert resolved.backend_name == "torchtune"
        assert resolved.framework == "torch"
        assert resolved.container_spec.command == ["tune", "run"]

    def test_resolve_with_default_backend(self, sft_config):
        BackendRegistry.register(TorchTuneBackend())
        trainer = LLMTrainer(config=sft_config)
        resolved = trainer.resolve()
        assert resolved.backend_name == "torchtune"

    def test_resolve_trl_dpo(self, dpo_config):
        BackendRegistry.register(TRLBackend())
        dpo_config.backend_name = "trl"
        trainer = LLMTrainer(config=dpo_config)
        resolved = trainer.resolve()
        assert resolved.backend_name == "trl"
        assert "trl" in resolved.packages_to_install[0]

    def test_resolve_unsloth_sft(self, sft_config):
        BackendRegistry.register(UnslothBackend())
        sft_config.backend_name = "unsloth"
        trainer = LLMTrainer(config=sft_config)
        resolved = trainer.resolve()
        assert resolved.backend_name == "unsloth"

    def test_resolve_unknown_backend_raises(self, sft_config):
        sft_config.backend_name = "nonexistent"
        trainer = LLMTrainer(config=sft_config)
        with pytest.raises(KeyError, match="No LLM backend registered"):
            trainer.resolve()

    def test_resolve_method_incompatibility(self, sft_config):
        BackendRegistry.register(TorchTuneBackend())
        sft_config.backend_name = "torchtune"
        sft_config.method = FineTuningMethod.PPO
        trainer = LLMTrainer(config=sft_config)
        with pytest.raises(ValueError, match="does not support PPO"):
            trainer.resolve()

    def test_resolve_merges_packages(self, sft_config):
        BackendRegistry.register(TRLBackend())
        sft_config.backend_name = "trl"
        trainer = LLMTrainer(
            config=sft_config,
            packages_to_install=["custom-package>=1.0"],
        )
        resolved = trainer.resolve()
        # Should have TRL's packages + user's custom package.
        assert any("trl" in p for p in resolved.packages_to_install)
        assert "custom-package>=1.0" in resolved.packages_to_install

    def test_resolve_deduplicates_packages(self, sft_config):
        BackendRegistry.register(TRLBackend())
        sft_config.backend_name = "trl"
        trainer = LLMTrainer(
            config=sft_config,
            packages_to_install=["trl>=0.12.0"],  # duplicate of backend's
        )
        resolved = trainer.resolve()
        trl_count = sum(1 for p in resolved.packages_to_install if p == "trl>=0.12.0")
        assert trl_count == 1

    def test_resolve_propagates_resources(self, sft_config):
        BackendRegistry.register(TorchTuneBackend())
        sft_config.backend_name = "torchtune"
        sft_config.num_nodes = 2
        trainer = LLMTrainer(
            config=sft_config,
            resources_per_node={"gpu": 4, "memory": "128Gi"},
        )
        resolved = trainer.resolve()
        assert resolved.resources_per_node == {"gpu": 4, "memory": "128Gi"}
        assert resolved.num_nodes == 2


# =========================================================================
# TrainJob spec generation
# =========================================================================


class TestTrainJobSpec:
    """Tests for the TrainJob spec generation integration."""

    def test_build_spec_torchtune(self, sft_config):
        BackendRegistry.register(TorchTuneBackend())
        sft_config.backend_name = "torchtune"
        trainer = LLMTrainer(
            config=sft_config,
            resources_per_node={"gpu": 2},
        )
        resolved = trainer.resolve()
        spec = build_trainjob_spec("my-sft-job", resolved)

        assert spec.name == "my-sft-job"
        assert spec.trainer_image.endswith("torchtune-trainer:latest")
        assert spec.trainer_command == ["tune", "run"]
        assert spec.num_nodes == 1
        assert spec.resources_per_node == {"gpu": 2}
        assert spec.labels["trainer.kubeflow.org/llm-backend"] == "torchtune"

    def test_build_spec_trl(self, dpo_config):
        BackendRegistry.register(TRLBackend())
        dpo_config.backend_name = "trl"
        trainer = LLMTrainer(config=dpo_config, resources_per_node={"gpu": 2})
        resolved = trainer.resolve()
        spec = build_trainjob_spec("my-dpo-job", resolved)

        assert spec.labels["trainer.kubeflow.org/llm-backend"] == "trl"
        assert len(spec.packages_to_install) > 0

    def test_build_spec_auto_runtime(self, sft_config):
        BackendRegistry.register(TorchTuneBackend())
        sft_config.backend_name = "torchtune"
        trainer = LLMTrainer(config=sft_config)
        resolved = trainer.resolve()
        spec = build_trainjob_spec("test-job", resolved)

        # When no runtime_name is given, it's auto-discovered.
        assert spec.runtime_ref == "auto:torch"

    def test_build_spec_explicit_runtime(self, sft_config):
        BackendRegistry.register(TorchTuneBackend())
        sft_config.backend_name = "torchtune"
        trainer = LLMTrainer(config=sft_config)
        resolved = trainer.resolve()
        spec = build_trainjob_spec(
            "test-job", resolved, runtime_name="torchtune-llama3.2-1b"
        )
        assert spec.runtime_ref == "torchtune-llama3.2-1b"

    def test_to_dict_structure(self, sft_config):
        BackendRegistry.register(TorchTuneBackend())
        sft_config.backend_name = "torchtune"
        trainer = LLMTrainer(
            config=sft_config,
            resources_per_node={"gpu": 1},
        )
        resolved = trainer.resolve()
        spec = build_trainjob_spec("test-job", resolved)
        d = spec.to_dict()

        assert d["apiVersion"] == "trainer.kubeflow.org/v1"
        assert d["kind"] == "TrainJob"
        assert d["metadata"]["name"] == "test-job"
        assert "trainer" in d["spec"]
        assert "runtimeRef" in d["spec"]
        assert d["spec"]["trainer"]["numNodes"] == 1

    def test_to_dict_trl_includes_packages(self, dpo_config):
        """TRL packages should be serialized in the TrainJob spec."""
        BackendRegistry.register(TRLBackend())
        dpo_config.backend_name = "trl"
        trainer = LLMTrainer(config=dpo_config)
        resolved = trainer.resolve()
        spec = build_trainjob_spec("trl-job", resolved)
        d = spec.to_dict()

        assert "packagesToInstall" in d["spec"]["trainer"]
        packages = d["spec"]["trainer"]["packagesToInstall"]
        assert any("trl" in p for p in packages)

    def test_to_dict_torchtune_no_packages(self, sft_config):
        """TorchTune has no extra packages — field should be absent."""
        BackendRegistry.register(TorchTuneBackend())
        sft_config.backend_name = "torchtune"
        trainer = LLMTrainer(config=sft_config)
        resolved = trainer.resolve()
        spec = build_trainjob_spec("tt-job", resolved)
        d = spec.to_dict()

        assert "packagesToInstall" not in d["spec"]["trainer"]

    def test_full_pipeline_trl_dpo(self, dpo_config):
        """End-to-end: LLMConfig → resolve() → TrainJobSpec → YAML dict."""
        BackendRegistry.register(TRLBackend())
        dpo_config.backend_name = "trl"
        trainer = LLMTrainer(config=dpo_config, resources_per_node={"gpu": 2})
        resolved = trainer.resolve()
        spec = build_trainjob_spec("e2e-dpo", resolved, runtime_name="trl-distributed")
        d = spec.to_dict()

        assert d["metadata"]["name"] == "e2e-dpo"
        assert d["spec"]["runtimeRef"]["name"] == "trl-distributed"
        assert d["spec"]["trainer"]["image"].endswith("trl-trainer:latest")
        assert d["spec"]["trainer"]["args"] == ["dpo"]
        env_names = [e["name"] for e in d["spec"]["trainer"]["env"]]
        assert "TRL_TRAINING_ARGS" in env_names
        assert d["spec"]["trainer"]["resourcesPerNode"] == {"gpu": 2}


# =========================================================================
# Cross-backend switching (the key value proposition)
# =========================================================================


class TestCrossBackendSwitching:
    """Test that users can switch backends by changing a single field.

    This is the core value proposition of the Dynamic LLM Trainer Framework:
    same model, same dataset, different backend — one line change.
    """

    @pytest.fixture(autouse=True)
    def _register(self, register_all_backends):
        pass

    def test_same_config_different_backends(self):
        """SFT on the same model/dataset, three different backends."""
        for backend_name in ["torchtune", "trl", "unsloth"]:
            config = LLMConfig(
                model="meta-llama/Llama-3.2-1B-Instruct",
                dataset="tatsu-lab/alpaca",
                method=FineTuningMethod.SFT,
                backend_name=backend_name,
            )
            trainer = LLMTrainer(config=config)
            resolved = trainer.resolve()
            assert resolved.backend_name == backend_name
            assert resolved.container_spec.image != ""
            assert len(resolved.container_spec.command) > 0

    def test_dpo_trl_vs_torchtune(self):
        """DPO is supported by both TRL and TorchTune — user picks."""
        for backend_name in ["trl", "torchtune"]:
            config = LLMConfig(
                model="meta-llama/Llama-3.2-1B-Instruct",
                dataset="trl-lib/ultrafeedback_binarized",
                method=FineTuningMethod.DPO,
                backend_name=backend_name,
                lora_config={"r": 16, "lora_alpha": 32},
            )
            trainer = LLMTrainer(config=config)
            resolved = trainer.resolve()
            assert resolved.backend_name == backend_name

    def test_ppo_only_trl(self):
        """PPO is only supported by TRL — others should fail."""
        config = LLMConfig(
            model="test/model",
            dataset="test/data",
            method=FineTuningMethod.PPO,
            backend_name="trl",
            extra_args={"reward_model": "test/reward-model"},
        )
        trainer = LLMTrainer(config=config)
        resolved = trainer.resolve()
        assert resolved.backend_name == "trl"

        # TorchTune should reject PPO.
        config.backend_name = "torchtune"
        trainer = LLMTrainer(config=config)
        with pytest.raises(ValueError):
            trainer.resolve()

        # Unsloth should reject PPO.
        config.backend_name = "unsloth"
        trainer = LLMTrainer(config=config)
        with pytest.raises(ValueError):
            trainer.resolve()

    def test_unsloth_rejects_multi_node_via_resolve(self):
        """Unsloth multi-node rejection should work through resolve(), not just validate()."""
        config = LLMConfig(
            model="test/model",
            dataset="test/data",
            method=FineTuningMethod.SFT,
            backend_name="unsloth",
            num_nodes=2,
        )
        trainer = LLMTrainer(config=config)
        with pytest.raises(ValueError, match="single-GPU"):
            trainer.resolve()

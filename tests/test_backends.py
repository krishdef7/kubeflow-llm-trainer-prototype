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

"""Tests for in-tree LLM backends: TorchTune, TRL, Unsloth."""

from __future__ import annotations

import json

import pytest

from kubeflow_llm_trainer import FineTuningMethod, LLMConfig, DataType
from kubeflow_llm_trainer.backends.torchtune import TorchTuneBackend
from kubeflow_llm_trainer.backends.trl import TRLBackend
from kubeflow_llm_trainer.backends.unsloth import UnslothBackend


# =========================================================================
# TorchTune Backend
# =========================================================================


class TestTorchTuneBackend:
    """Tests for the TorchTune backend."""

    @pytest.fixture()
    def backend(self) -> TorchTuneBackend:
        return TorchTuneBackend()

    def test_name(self, backend):
        assert backend.name == "torchtune"

    def test_framework_is_torch(self, backend):
        assert backend.framework == "torch"

    def test_supported_methods(self, backend):
        assert FineTuningMethod.SFT in backend.supported_methods
        assert FineTuningMethod.LORA in backend.supported_methods
        assert FineTuningMethod.QLORA in backend.supported_methods
        assert FineTuningMethod.DPO in backend.supported_methods
        assert FineTuningMethod.PPO not in backend.supported_methods

    def test_validate_rejects_ppo(self, backend, sft_config):
        sft_config.method = FineTuningMethod.PPO
        with pytest.raises(ValueError, match="does not support PPO"):
            backend.validate(sft_config)

    def test_validate_rejects_orpo(self, backend, sft_config):
        sft_config.method = FineTuningMethod.ORPO
        with pytest.raises(ValueError, match="does not support ORPO"):
            backend.validate(sft_config)

    def test_validate_rejects_kto(self, backend, sft_config):
        sft_config.method = FineTuningMethod.KTO
        with pytest.raises(ValueError, match="does not support KTO"):
            backend.validate(sft_config)

    def test_validate_rejects_unknown_lora_keys(self, backend):
        config = LLMConfig(
            model="test/model",
            dataset="test/data",
            method=FineTuningMethod.LORA,
            lora_config={"r": 8, "unknown_key": True},
        )
        with pytest.raises(ValueError, match="Unknown TorchTune LoRA"):
            backend.validate(config)

    def test_validate_accepts_valid_lora(self, backend, lora_config):
        # Should not raise.
        backend.validate(lora_config)

    def test_container_spec_sft(self, backend, sft_config):
        spec = backend.to_container_spec(sft_config)
        assert spec.command == ["tune", "run"]
        assert spec.image.endswith("torchtune-trainer:latest")
        assert "full_finetune_single_device" in spec.args
        # Should have --config and key=value overrides.
        assert "--config" in spec.args
        assert f"epochs={sft_config.epochs}" in spec.args

    def test_container_spec_lora(self, backend, lora_config):
        spec = backend.to_container_spec(lora_config)
        assert "lora_finetune_single_device" in spec.args
        # LoRA args use TorchTune's top-level key names, not a nested namespace.
        assert any(a.startswith("lora_rank=") for a in spec.args)
        assert any(a.startswith("lora_alpha=") for a in spec.args)

    def test_container_spec_distributed(self, backend):
        config = LLMConfig(
            model="test/model",
            dataset="test/data",
            method=FineTuningMethod.SFT,
            num_nodes=2,
        )
        spec = backend.to_container_spec(config)
        assert "full_finetune_distributed" in spec.args

    def test_container_spec_includes_env(self, backend, sft_config):
        spec = backend.to_container_spec(sft_config)
        assert spec.env["KUBEFLOW_MODEL_PATH"] == "/mnt/model"
        assert spec.env["KUBEFLOW_DATASET_PATH"] == "/mnt/dataset"
        # The model path is also injected as an OmegaConf override so that
        # TorchTune's checkpointer reads from the Kubeflow PVC mount.
        assert "checkpointer.checkpoint_dir=/mnt/model" in spec.args
        assert "checkpointer.output_dir=/mnt/output" in spec.args

    def test_container_spec_qlora(self, backend):
        """QLoRA should use the qlora_finetune_* recipe, not quantized_*."""
        config = LLMConfig(
            model="test/model",
            dataset="test/data",
            method=FineTuningMethod.QLORA,
        )
        spec = backend.to_container_spec(config)
        assert "qlora_finetune_single_device" in spec.args

    def test_container_spec_torchtune_config_override(self, backend):
        """When torchtune_config is provided, it overrides the base config."""
        config = LLMConfig(
            model="meta-llama/Llama-3.2-1B-Instruct",
            dataset="tatsu-lab/alpaca",
            method=FineTuningMethod.LORA,
            extra_args={"torchtune_config": "llama3_2/1B_lora_single_device"},
            lora_config={"r": 8, "lora_alpha": 16},
        )
        spec = backend.to_container_spec(config)
        # The recipe is determined by method + is_distributed.
        assert spec.args[0] == "lora_finetune_single_device"
        # But the --config uses the model-specific override.
        config_idx = spec.args.index("--config")
        assert spec.args[config_idx + 1] == "llama3_2/1B_lora_single_device"

    def test_container_spec_custom_image(self, backend):
        config = LLMConfig(
            model="test/model",
            dataset="test/data",
            extra_args={"image": "custom-image:v1"},
        )
        spec = backend.to_container_spec(config)
        assert spec.image == "custom-image:v1"

    def test_container_spec_hyperparams(self, backend):
        config = LLMConfig(
            model="test/model",
            dataset="test/data",
            dtype=DataType.FP16,
            epochs=5,
            batch_size=8,
            learning_rate=1e-4,
        )
        spec = backend.to_container_spec(config)
        assert "dtype=fp16" in spec.args
        assert "epochs=5" in spec.args
        assert "batch_size=8" in spec.args

    def test_repr(self, backend):
        repr_str = repr(backend)
        assert "TorchTuneBackend" in repr_str
        assert "torchtune" in repr_str

    def test_supported_methods_includes_full(self, backend):
        """FULL fine-tuning must be in TorchTune's supported methods."""
        assert FineTuningMethod.FULL in backend.supported_methods

    def test_container_spec_lora_bool_value(self, backend):
        """Bool LoRA config values should serialize as 'true'/'false'."""
        config = LLMConfig(
            model="test/model",
            dataset="test/data",
            method=FineTuningMethod.LORA,
            lora_config={"r": 8, "apply_lora_to_mlp": True},
        )
        spec = backend.to_container_spec(config)
        assert "apply_lora_to_mlp=true" in spec.args

    def test_container_spec_lora_list_value(self, backend):
        """List LoRA config values should serialize as OmegaConf list syntax."""
        config = LLMConfig(
            model="test/model",
            dataset="test/data",
            method=FineTuningMethod.LORA,
            lora_config={"r": 8, "lora_attn_modules": ["q_proj", "v_proj"]},
        )
        spec = backend.to_container_spec(config)
        assert "lora_attn_modules=[q_proj,v_proj]" in spec.args

    def test_container_spec_learning_rate_override(self, backend):
        """Learning rate should map to optimizer.lr= OmegaConf override."""
        config = LLMConfig(
            model="test/model", dataset="test/data",
            learning_rate=1e-4,
        )
        spec = backend.to_container_spec(config)
        assert "optimizer.lr=0.0001" in spec.args


# =========================================================================
# TRL Backend
# =========================================================================


class TestTRLBackend:
    """Tests for the TRL backend."""

    @pytest.fixture()
    def backend(self) -> TRLBackend:
        return TRLBackend()

    def test_name(self, backend):
        assert backend.name == "trl"

    def test_framework_is_torch(self, backend):
        assert backend.framework == "torch"

    def test_packages_include_trl(self, backend):
        assert any("trl" in p for p in backend.packages_to_install)
        assert any("peft" in p for p in backend.packages_to_install)
        assert any("accelerate" in p for p in backend.packages_to_install)

    def test_supported_methods(self, backend):
        assert FineTuningMethod.SFT in backend.supported_methods
        assert FineTuningMethod.DPO in backend.supported_methods
        assert FineTuningMethod.PPO in backend.supported_methods
        assert FineTuningMethod.ORPO in backend.supported_methods
        assert FineTuningMethod.KTO in backend.supported_methods

    def test_validate_rejects_full(self, backend, sft_config):
        sft_config.method = FineTuningMethod.FULL
        with pytest.raises(ValueError, match="does not have a 'full'"):
            backend.validate(sft_config)

    def test_validate_rejects_qlora_method(self, backend, sft_config):
        sft_config.method = FineTuningMethod.QLORA
        with pytest.raises(ValueError, match="QLoRA with TRL"):
            backend.validate(sft_config)

    def test_validate_rejects_lora_method(self, backend, sft_config):
        sft_config.method = FineTuningMethod.LORA
        with pytest.raises(ValueError, match="LoRA with TRL"):
            backend.validate(sft_config)

    def test_validate_rejects_unknown_lora_keys(self, backend):
        config = LLMConfig(
            model="test/model",
            dataset="test/data",
            method=FineTuningMethod.SFT,
            lora_config={"r": 8, "bad_key": True},
        )
        with pytest.raises(ValueError, match="Unknown TRL/PEFT LoRA"):
            backend.validate(config)

    def test_container_spec_sft(self, backend, sft_config):
        spec = backend.to_container_spec(sft_config)
        assert spec.command == [
            "python", "-m", "kubeflow_llm_trainer.entrypoints.trl_runner"
        ]
        assert spec.args == ["sft"]
        assert "TRL_TRAINING_ARGS" in spec.env

    def test_container_spec_dpo(self, backend, dpo_config):
        spec = backend.to_container_spec(dpo_config)
        assert spec.args == ["dpo"]
        training_args = json.loads(spec.env["TRL_TRAINING_ARGS"])
        assert training_args["beta"] == 0.1

    def test_validate_rejects_ppo_without_reward_model(self, backend):
        """PPO requires reward_model in extra_args — caught at validate time."""
        config = LLMConfig(
            model="test/model",
            dataset="test/data",
            method=FineTuningMethod.PPO,
        )
        with pytest.raises(ValueError, match="reward_model"):
            backend.validate(config)

    def test_container_spec_ppo_with_reward_model(self, backend):
        config = LLMConfig(
            model="test/model",
            dataset="test/data",
            method=FineTuningMethod.PPO,
            extra_args={"reward_model": "OpenAssistant/reward-model-deberta-v3"},
        )
        spec = backend.to_container_spec(config)
        assert spec.args == ["ppo"]
        training_args = json.loads(spec.env["TRL_TRAINING_ARGS"])
        assert training_args["reward_model"] != ""

    def test_container_spec_training_args_content(self, backend, sft_config):
        spec = backend.to_container_spec(sft_config)
        args = json.loads(spec.env["TRL_TRAINING_ARGS"])
        assert args["model_name_or_path"] == sft_config.model
        assert args["dataset_name"] == sft_config.dataset
        assert args["num_train_epochs"] == sft_config.epochs
        assert args["per_device_train_batch_size"] == sft_config.batch_size
        assert args["learning_rate"] == sft_config.learning_rate
        assert args["bf16"] is True  # default dtype is BF16

    def test_container_spec_lora_args(self, backend):
        config = LLMConfig(
            model="test/model",
            dataset="test/data",
            method=FineTuningMethod.SFT,
            lora_config={"r": 16, "lora_alpha": 32, "lora_dropout": 0.05},
        )
        spec = backend.to_container_spec(config)
        args = json.loads(spec.env["TRL_TRAINING_ARGS"])
        assert args["use_peft"] is True
        assert args["lora_r"] == 16
        assert args["lora_alpha"] == 32
        assert args["lora_dropout"] == 0.05

    def test_container_spec_qlora_via_lora_config(self, backend):
        config = LLMConfig(
            model="test/model",
            dataset="test/data",
            method=FineTuningMethod.SFT,
            lora_config={"r": 16, "quantize_base": True},
        )
        spec = backend.to_container_spec(config)
        args = json.loads(spec.env["TRL_TRAINING_ARGS"])
        assert args["load_in_4bit"] is True
        assert args["bnb_4bit_quant_type"] == "nf4"

    def test_container_spec_kto(self, backend):
        config = LLMConfig(
            model="test/model",
            dataset="test/data",
            method=FineTuningMethod.KTO,
            extra_args={"desirable_weight": 2.0, "undesirable_weight": 0.5},
        )
        spec = backend.to_container_spec(config)
        assert spec.args == ["kto"]
        args = json.loads(spec.env["TRL_TRAINING_ARGS"])
        assert args["desirable_weight"] == 2.0
        assert args["undesirable_weight"] == 0.5

    def test_container_spec_orpo(self, backend):
        """ORPO should produce correct TRL command and include orpo_alpha."""
        config = LLMConfig(
            model="test/model",
            dataset="test/data",
            method=FineTuningMethod.ORPO,
            extra_args={"orpo_alpha": 0.5},
        )
        spec = backend.to_container_spec(config)
        assert spec.args == ["orpo"]
        args = json.loads(spec.env["TRL_TRAINING_ARGS"])
        assert args["orpo_alpha"] == 0.5

    def test_container_spec_orpo_default_alpha(self, backend):
        """ORPO without explicit orpo_alpha should use default 0.1."""
        config = LLMConfig(
            model="test/model",
            dataset="test/data",
            method=FineTuningMethod.ORPO,
        )
        spec = backend.to_container_spec(config)
        args = json.loads(spec.env["TRL_TRAINING_ARGS"])
        assert args["orpo_alpha"] == 0.1

    def test_container_spec_ppo_reward_model_value(self, backend):
        """PPO reward_model should contain the exact value specified."""
        config = LLMConfig(
            model="test/model",
            dataset="test/data",
            method=FineTuningMethod.PPO,
            extra_args={"reward_model": "OpenAssistant/reward-model-deberta-v3"},
        )
        spec = backend.to_container_spec(config)
        args = json.loads(spec.env["TRL_TRAINING_ARGS"])
        assert args["reward_model"] == "OpenAssistant/reward-model-deberta-v3"

    def test_container_spec_lora_target_modules(self, backend):
        """target_modules should map to lora_target_modules in args JSON."""
        config = LLMConfig(
            model="test/model",
            dataset="test/data",
            method=FineTuningMethod.SFT,
            lora_config={"r": 16, "target_modules": ["q_proj", "v_proj"]},
        )
        spec = backend.to_container_spec(config)
        args = json.loads(spec.env["TRL_TRAINING_ARGS"])
        assert args["lora_target_modules"] == ["q_proj", "v_proj"]

    def test_repr(self, backend):
        repr_str = repr(backend)
        assert "TRLBackend" in repr_str
        assert "trl" in repr_str


# =========================================================================
# Unsloth Backend
# =========================================================================


class TestUnslothBackend:
    """Tests for the Unsloth backend."""

    @pytest.fixture()
    def backend(self) -> UnslothBackend:
        return UnslothBackend()

    def test_name(self, backend):
        assert backend.name == "unsloth"

    def test_packages_include_unsloth(self, backend):
        assert any("unsloth" in p for p in backend.packages_to_install)

    def test_supported_methods(self, backend):
        assert FineTuningMethod.SFT in backend.supported_methods
        assert FineTuningMethod.DPO in backend.supported_methods
        assert FineTuningMethod.LORA in backend.supported_methods
        assert FineTuningMethod.PPO not in backend.supported_methods

    def test_validate_rejects_full(self, backend, sft_config):
        sft_config.method = FineTuningMethod.FULL
        with pytest.raises(ValueError, match="does not support full"):
            backend.validate(sft_config)

    def test_validate_rejects_ppo(self, backend, sft_config):
        sft_config.method = FineTuningMethod.PPO
        with pytest.raises(ValueError, match="does not support PPO"):
            backend.validate(sft_config)

    def test_validate_rejects_multi_node(self, backend):
        config = LLMConfig(
            model="test/model",
            dataset="test/data",
            method=FineTuningMethod.SFT,
            num_nodes=4,
        )
        with pytest.raises(ValueError, match="single-GPU training only"):
            backend.validate(config)

    def test_validate_rejects_unknown_lora_keys(self, backend):
        config = LLMConfig(
            model="test/model",
            dataset="test/data",
            method=FineTuningMethod.LORA,
            lora_config={"r": 8, "bad_key": True},
        )
        with pytest.raises(ValueError, match="Unknown Unsloth LoRA"):
            backend.validate(config)

    def test_container_spec_sft(self, backend, sft_config):
        spec = backend.to_container_spec(sft_config)
        assert "unsloth_runner" in spec.command[-1]
        assert "UNSLOTH_TRAINING_CONFIG" in spec.env
        assert spec.env["UNSLOTH_TRAINER_TYPE"] == "sft"

    def test_container_spec_dpo(self, backend, dpo_config):
        spec = backend.to_container_spec(dpo_config)
        assert spec.env["UNSLOTH_TRAINER_TYPE"] == "dpo"

    def test_container_spec_config_content(self, backend, sft_config):
        spec = backend.to_container_spec(sft_config)
        cfg = json.loads(spec.env["UNSLOTH_TRAINING_CONFIG"])
        assert cfg["model_name"] == sft_config.model
        assert cfg["dataset_name"] == sft_config.dataset
        assert cfg["num_train_epochs"] == sft_config.epochs
        assert cfg["per_device_train_batch_size"] == sft_config.batch_size

    def test_container_spec_qlora(self, backend):
        config = LLMConfig(
            model="test/model",
            dataset="test/data",
            method=FineTuningMethod.QLORA,
        )
        spec = backend.to_container_spec(config)
        cfg = json.loads(spec.env["UNSLOTH_TRAINING_CONFIG"])
        assert cfg["load_in_4bit"] is True

    def test_container_spec_lora_defaults(self, backend, sft_config):
        spec = backend.to_container_spec(sft_config)
        cfg = json.loads(spec.env["UNSLOTH_TRAINING_CONFIG"])
        # Defaults when no lora_config is provided.
        assert cfg["lora_r"] == 16
        assert cfg["lora_alpha"] == 16
        assert cfg["use_gradient_checkpointing"] == "unsloth"

    def test_container_spec_custom_lora(self, backend):
        config = LLMConfig(
            model="test/model",
            dataset="test/data",
            method=FineTuningMethod.LORA,
            lora_config={"r": 32, "lora_alpha": 64, "use_rslora": True},
        )
        spec = backend.to_container_spec(config)
        cfg = json.loads(spec.env["UNSLOTH_TRAINING_CONFIG"])
        assert cfg["lora_r"] == 32
        assert cfg["lora_alpha"] == 64
        assert cfg["use_rslora"] is True

    def test_validate_rejects_kto(self, backend, sft_config):
        """Unsloth explicitly rejects KTO."""
        sft_config.method = FineTuningMethod.KTO
        with pytest.raises(ValueError, match="does not support KTO"):
            backend.validate(sft_config)

    def test_supported_methods_includes_orpo(self, backend):
        """Unsloth supports ORPO — must be in supported_methods."""
        assert FineTuningMethod.ORPO in backend.supported_methods

    def test_container_spec_orpo(self, backend):
        """ORPO should produce trainer_type='orpo'."""
        config = LLMConfig(
            model="test/model",
            dataset="test/data",
            method=FineTuningMethod.ORPO,
        )
        spec = backend.to_container_spec(config)
        assert spec.env["UNSLOTH_TRAINER_TYPE"] == "orpo"

    def test_container_spec_command_full(self, backend, sft_config):
        """Full command assertion instead of substring check."""
        spec = backend.to_container_spec(sft_config)
        assert spec.command == [
            "python", "-m", "kubeflow_llm_trainer.entrypoints.unsloth_runner"
        ]

    def test_repr(self, backend):
        repr_str = repr(backend)
        assert "UnslothBackend" in repr_str
        assert "unsloth" in repr_str

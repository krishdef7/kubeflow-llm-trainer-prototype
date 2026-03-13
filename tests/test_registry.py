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

"""Tests for BackendRegistry."""

from __future__ import annotations

import pytest

from kubeflow_llm_trainer import BackendRegistry, LLMBackend, LLMConfig, ContainerSpec
from kubeflow_llm_trainer.backends.torchtune import TorchTuneBackend
from kubeflow_llm_trainer.backends.trl import TRLBackend
from kubeflow_llm_trainer.backends.unsloth import UnslothBackend


class TestBackendRegistration:
    """Test backend registration via all supported paths."""

    def test_register_instance(self):
        backend = TorchTuneBackend()
        BackendRegistry.register(backend)
        assert BackendRegistry.is_registered("torchtune")
        assert BackendRegistry.get("torchtune") is backend

    def test_register_class_decorator(self):
        @BackendRegistry.register
        class MockBackend(LLMBackend):
            @property
            def name(self) -> str:
                return "mock"

            def validate(self, config: LLMConfig) -> None:
                pass

            def to_container_spec(self, config: LLMConfig) -> ContainerSpec:
                return ContainerSpec(image="mock:latest", command=["echo"])

        assert BackendRegistry.is_registered("mock")

    def test_register_rejects_non_backend(self):
        with pytest.raises(TypeError, match="Expected an LLMBackend"):
            BackendRegistry.register("not_a_backend")  # type: ignore

    def test_register_rejects_non_subclass(self):
        with pytest.raises(TypeError, match="Expected an LLMBackend subclass"):
            BackendRegistry.register(str)  # type: ignore

    def test_explicit_registration_overrides_entry_points(self):
        """Explicitly registered backends take precedence over entry points."""
        custom = TorchTuneBackend()
        BackendRegistry.register(custom)
        assert BackendRegistry.get("torchtune") is custom


class TestBackendRetrieval:
    """Test backend retrieval and error handling."""

    def test_get_registered_backend(self):
        BackendRegistry.register(TRLBackend())
        assert BackendRegistry.get("trl").name == "trl"

    def test_get_unknown_backend_raises_keyerror(self):
        with pytest.raises(KeyError, match="No LLM backend registered"):
            BackendRegistry.get("nonexistent")

    def test_get_default_returns_torchtune(self, register_all_backends):
        default = BackendRegistry.get_default()
        assert default.name == "torchtune"

    def test_get_default_falls_back_to_alphabetical(self):
        BackendRegistry.register(UnslothBackend())
        BackendRegistry.register(TRLBackend())
        # No torchtune registered → falls back to first alphabetically.
        default = BackendRegistry.get_default()
        assert default.name == "trl"  # "trl" < "unsloth"

    def test_get_default_empty_registry_raises(self):
        with pytest.raises(RuntimeError, match="No LLM backends registered"):
            BackendRegistry.get_default()


class TestBackendListing:
    """Test listing and inspection."""

    def test_list_backends_empty(self):
        assert BackendRegistry.list_backends() == []

    def test_list_backends_sorted(self, register_all_backends):
        names = BackendRegistry.list_backends()
        assert names == ["torchtune", "trl", "unsloth"]

    def test_is_registered_true(self):
        BackendRegistry.register(TRLBackend())
        assert BackendRegistry.is_registered("trl")

    def test_is_registered_false(self):
        assert not BackendRegistry.is_registered("trl")


class TestRegistryReset:
    """Test the _reset method for test isolation."""

    def test_reset_clears_all(self, register_all_backends):
        assert len(BackendRegistry.list_backends()) == 3
        BackendRegistry._reset()
        assert len(BackendRegistry.list_backends()) == 0

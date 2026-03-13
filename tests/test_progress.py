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

"""Tests for Kubeflow Trainer progress reporting (KEP-2779).

These tests verify the SDK-side client for the status server implemented
in kubeflow/trainer#3227.  They test the client logic without requiring
a running status server — HTTP calls are mocked.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from kubeflow_llm_trainer.progress import (
    ENV_STATUS_TOKEN,
    ENV_STATUS_URL,
    KubeflowProgressReporter,
    KubeflowTrainerCallback,
    Metric,
    TrainerStatus,
    is_progress_reporting_available,
)


# =========================================================================
# TrainerStatus payload serialization
# =========================================================================


class TestTrainerStatus:
    """Tests for the TrainerStatus data type and serialization."""

    def test_empty_status(self):
        status = TrainerStatus()
        payload = status.to_payload()
        assert payload == {"trainerStatus": {}}

    def test_progress_only(self):
        status = TrainerStatus(progress=50)
        payload = status.to_payload()
        assert payload == {"trainerStatus": {"progress": 50}}

    def test_full_status(self):
        status = TrainerStatus(
            progress=75,
            eta="2m 30s",
            metrics=[
                Metric(name="loss", value="0.42"),
                Metric(name="accuracy", value="0.91"),
            ],
        )
        payload = status.to_payload()
        assert payload["trainerStatus"]["progress"] == 75
        assert payload["trainerStatus"]["eta"] == "2m 30s"
        assert len(payload["trainerStatus"]["metrics"]) == 2
        assert payload["trainerStatus"]["metrics"][0] == {
            "name": "loss",
            "value": "0.42",
        }

    def test_payload_is_json_serializable(self):
        status = TrainerStatus(
            progress=100,
            metrics=[Metric(name="loss", value="0.01")],
        )
        # Must not raise.
        serialized = json.dumps(status.to_payload())
        assert '"progress": 100' in serialized


# =========================================================================
# is_progress_reporting_available()
# =========================================================================


class TestProgressAvailability:
    """Tests for environment variable detection."""

    def test_available_when_url_set(self, monkeypatch):
        monkeypatch.setenv(ENV_STATUS_URL, "https://controller:8443/status")
        assert is_progress_reporting_available() is True

    def test_not_available_when_url_missing(self, monkeypatch):
        monkeypatch.delenv(ENV_STATUS_URL, raising=False)
        assert is_progress_reporting_available() is False

    def test_not_available_when_url_empty(self, monkeypatch):
        monkeypatch.setenv(ENV_STATUS_URL, "")
        assert is_progress_reporting_available() is False


# =========================================================================
# KubeflowProgressReporter
# =========================================================================


class TestKubeflowProgressReporter:
    """Tests for the low-level HTTP client."""

    def test_disabled_when_env_not_set(self, monkeypatch):
        monkeypatch.delenv(ENV_STATUS_URL, raising=False)
        reporter = KubeflowProgressReporter()
        assert reporter.enabled is False
        # Should return True (no-op) without making any HTTP call.
        assert reporter.report(TrainerStatus(progress=50)) is True

    def test_enabled_when_env_set(self, monkeypatch):
        monkeypatch.setenv(ENV_STATUS_URL, "https://controller:8443/status")
        monkeypatch.setenv(ENV_STATUS_TOKEN, "test-token")
        reporter = KubeflowProgressReporter()
        assert reporter.enabled is True

    @patch("kubeflow_llm_trainer.progress.urlopen")
    def test_report_sends_correct_payload(self, mock_urlopen, monkeypatch):
        monkeypatch.setenv(ENV_STATUS_URL, "https://controller:8443/status")
        monkeypatch.setenv(ENV_STATUS_TOKEN, "jwt-token-here")

        # Mock a successful response.
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        reporter = KubeflowProgressReporter()
        result = reporter.report(TrainerStatus(
            progress=42,
            metrics=[Metric(name="loss", value="0.5")],
        ))

        assert result is True
        # Verify the request was made.
        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        assert request.get_method() == "POST"
        assert request.get_header("Authorization") == "Bearer jwt-token-here"
        assert request.get_header("Content-type") == "application/json"

        # Verify the payload.
        body = json.loads(request.data.decode("utf-8"))
        assert body["trainerStatus"]["progress"] == 42
        assert body["trainerStatus"]["metrics"][0]["name"] == "loss"

    @patch("kubeflow_llm_trainer.progress.urlopen")
    def test_report_handles_http_error(self, mock_urlopen, monkeypatch):
        from urllib.error import HTTPError
        monkeypatch.setenv(ENV_STATUS_URL, "https://controller:8443/status")
        mock_urlopen.side_effect = HTTPError(
            url="", code=500, msg="Internal", hdrs=None,  # type: ignore
            fp=MagicMock(read=MagicMock(return_value=b"server error")),
        )

        reporter = KubeflowProgressReporter()
        result = reporter.report(TrainerStatus(progress=10))
        assert result is False  # graceful failure, no exception


# =========================================================================
# KubeflowTrainerCallback — HuggingFace Trainer integration
# =========================================================================


@dataclass
class MockTrainerState:
    """Simulates HuggingFace's TrainerState."""
    global_step: int = 0
    max_steps: int = 100


class TestKubeflowTrainerCallback:
    """Tests for the HuggingFace TrainerCallback integration."""

    def test_callback_is_noop_when_disabled(self, monkeypatch):
        monkeypatch.delenv(ENV_STATUS_URL, raising=False)
        callback = KubeflowTrainerCallback()
        # None of these should raise.
        callback.on_train_begin(args=None, state=None, control=None)
        callback.on_log(args=None, state=None, control=None, logs={"loss": 0.5})
        callback.on_train_end(args=None, state=None, control=None)

    @patch("kubeflow_llm_trainer.progress.urlopen")
    def test_on_train_begin_reports_zero(self, mock_urlopen, monkeypatch):
        monkeypatch.setenv(ENV_STATUS_URL, "https://controller:8443/status")
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        callback = KubeflowTrainerCallback()
        callback.on_train_begin(args=None, state=None, control=None)

        # Verify 0% progress was reported.
        call_args = mock_urlopen.call_args
        body = json.loads(call_args[0][0].data.decode("utf-8"))
        assert body["trainerStatus"]["progress"] == 0

    @patch("kubeflow_llm_trainer.progress.urlopen")
    def test_on_log_reports_progress_and_metrics(self, mock_urlopen, monkeypatch):
        monkeypatch.setenv(ENV_STATUS_URL, "https://controller:8443/status")
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        callback = KubeflowTrainerCallback()
        callback._start_time = 1000.0  # fake start time

        state = MockTrainerState(global_step=50, max_steps=100)
        logs = {"loss": 0.42, "learning_rate": 2e-5}

        with patch("kubeflow_llm_trainer.progress.time") as mock_time:
            mock_time.time.return_value = 1100.0  # 100 seconds elapsed
            callback.on_log(args=None, state=state, control=None, logs=logs)

        body = json.loads(mock_urlopen.call_args[0][0].data.decode("utf-8"))
        assert body["trainerStatus"]["progress"] == 50
        assert "eta" in body["trainerStatus"]
        metrics = body["trainerStatus"]["metrics"]
        metric_names = [m["name"] for m in metrics]
        assert "loss" in metric_names
        assert "learning_rate" in metric_names

    @patch("kubeflow_llm_trainer.progress.urlopen")
    def test_on_train_end_reports_100(self, mock_urlopen, monkeypatch):
        monkeypatch.setenv(ENV_STATUS_URL, "https://controller:8443/status")
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        callback = KubeflowTrainerCallback()
        callback.on_train_end(args=None, state=None, control=None)

        body = json.loads(mock_urlopen.call_args[0][0].data.decode("utf-8"))
        assert body["trainerStatus"]["progress"] == 100

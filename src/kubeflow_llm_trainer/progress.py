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

"""SDK-side progress reporting for the Kubeflow Trainer status server.

This module provides the client-side integration for KEP-2779 (TrainJob
progress tracking), implemented in kubeflow/trainer#3227 by @robert-bell.

The Kubeflow Trainer controller injects three environment variables into
training pods via the ``trainjob-status`` plugin:

* ``KUBEFLOW_TRAINER_SERVER_URL``     — HTTPS endpoint to POST updates to.
* ``KUBEFLOW_TRAINER_SERVER_CA_CERT`` — **File path** to the PEM-encoded CA
  cert (mounted from a ConfigMap) to trust the controller's self-signed TLS.
* ``KUBEFLOW_TRAINER_SERVER_TOKEN``   — Projected ServiceAccount JWT for
  authenticating with the status server.  The token audience encodes the
  TrainJob namespace/name for authorization.

This module provides:

1. ``KubeflowProgressReporter`` — a standalone client that POSTs progress
   updates (progress %, ETA, custom metrics) to the status server.

2. ``KubeflowTrainerCallback`` — a HuggingFace Transformers
   ``TrainerCallback`` subclass that automatically reports progress on each
   logging step.  This works with any HF Trainer-based framework: TRL's
   ``SFTTrainer``, ``DPOTrainer``, ``PPOTrainer``, etc., and Unsloth
   (which patches TRL's trainers).

   Note: HuggingFace Transformers now ships an official ``KubeflowCallback``
   (merged via huggingface/transformers#44487) that uses the same env vars.
   Our callback is compatible and can coexist; it provides additional ETA
   estimation not present in the upstream version.

Integration with the Dynamic LLM Trainer Framework
---------------------------------------------------

The entrypoints (``trl_runner.py``, ``unsloth_runner.py``) automatically
inject ``KubeflowTrainerCallback`` into the trainer's callback list when
the status server env vars are detected.  No user action is required.

For TorchTune, progress reporting happens via a custom recipe hook that
calls ``KubeflowProgressReporter`` directly, since TorchTune does not
use the HuggingFace Trainer API.

See also: https://github.com/kubeflow/trainer/pull/3227
"""

from __future__ import annotations

import json
import logging
import os
import ssl
import time
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# Environment variables injected by the trainjob-status plugin.
# Ref: kubeflow/trainer#3227 pkg/runtime/framework/plugins/trainjobstatus/
# Names were finalized in the review by @astefanutti:
# https://github.com/kubeflow/trainer/pull/3227#discussion_r2148925508
ENV_STATUS_URL = "KUBEFLOW_TRAINER_SERVER_URL"
ENV_STATUS_CA_CERT = "KUBEFLOW_TRAINER_SERVER_CA_CERT"
ENV_STATUS_TOKEN = "KUBEFLOW_TRAINER_SERVER_TOKEN"


def is_progress_reporting_available() -> bool:
    """Check whether the status server env vars are present.

    Returns ``True`` when running inside a Kubeflow TrainJob pod with the
    ``TrainJobStatus`` feature gate enabled.
    """
    return bool(os.environ.get(ENV_STATUS_URL))


# ---------------------------------------------------------------------------
# Data types matching the TrainJob status API (KEP-2779)
# ---------------------------------------------------------------------------

@dataclass
class Metric:
    """A single training metric reported to the status server.

    Attributes:
        name: Metric name (e.g. ``"loss"``, ``"accuracy"``).
        value: Metric value as a string (Kubernetes API convention).
    """

    name: str
    value: str


@dataclass
class TrainerStatus:
    """Payload POSTed to the Kubeflow Trainer status server.

    This matches the ``TrainerStatus`` type defined in
    ``pkg/apis/trainer/v1alpha1/trainjob_types.go``.

    Attributes:
        progress: Training progress as a percentage (0–100).
        eta: Estimated time remaining as a human-readable string.
        metrics: List of custom training metrics.
    """

    progress: int | None = None
    eta: str | None = None
    metrics: list[Metric] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        """Serialize to the JSON payload the status server expects."""
        result: dict[str, Any] = {"trainerStatus": {}}
        ts = result["trainerStatus"]
        if self.progress is not None:
            ts["progress"] = self.progress
        if self.eta is not None:
            ts["eta"] = self.eta
        if self.metrics:
            ts["metrics"] = [asdict(m) for m in self.metrics]
        return result


# ---------------------------------------------------------------------------
# KubeflowProgressReporter — low-level HTTP client
# ---------------------------------------------------------------------------

class KubeflowProgressReporter:
    """Client for the Kubeflow Trainer status server.

    Reads configuration from environment variables.  If the env vars are
    not present (e.g. running outside a TrainJob, or the feature gate is
    disabled), all calls are silent no-ops.

    Usage::

        reporter = KubeflowProgressReporter()
        reporter.report(TrainerStatus(progress=50, metrics=[
            Metric(name="loss", value="0.42"),
        ]))

    Thread safety: this class is safe to call from multiple threads.
    """

    def __init__(self) -> None:
        self._url = os.environ.get(ENV_STATUS_URL, "")
        self._token = os.environ.get(ENV_STATUS_TOKEN, "")
        self._ssl_ctx = self._build_ssl_context()
        self._enabled = bool(self._url)

        if self._enabled:
            logger.info("Kubeflow progress reporting enabled: %s", self._url)
        else:
            logger.debug(
                "Kubeflow progress reporting disabled: %s not set.",
                ENV_STATUS_URL,
            )

    @property
    def enabled(self) -> bool:
        """Whether progress reporting is active."""
        return self._enabled

    def report(self, status: TrainerStatus) -> bool:
        """POST a status update to the Kubeflow Trainer status server.

        Returns ``True`` on success, ``False`` on failure (logged, not raised).
        Returns ``True`` immediately if reporting is disabled (no-op).
        """
        if not self._enabled:
            return True

        payload = json.dumps(status.to_payload()).encode("utf-8")
        request = Request(
            self._url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._token}",
            },
            method="POST",
        )

        try:
            with urlopen(request, context=self._ssl_ctx, timeout=10) as resp:
                if resp.status == 200:
                    logger.debug("Progress update sent: %d%%", status.progress)
                    return True
                logger.warning(
                    "Status server returned %d: %s",
                    resp.status,
                    resp.read().decode("utf-8", errors="replace"),
                )
                return False
        except HTTPError as e:
            logger.warning(
                "Progress update failed (HTTP %d): %s",
                e.code,
                e.read().decode("utf-8", errors="replace"),
            )
            return False
        except (URLError, OSError) as e:
            logger.warning("Progress update failed: %s", e)
            return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_ssl_context(self) -> ssl.SSLContext | None:
        """Build an SSL context trusting the controller's CA cert.

        ``KUBEFLOW_TRAINER_SERVER_CA_CERT`` contains the **file path** to a
        PEM-encoded CA certificate (mounted from a ConfigMap by the
        trainjob-status plugin), NOT the inline PEM content.

        Ref: robert-bell confirmed this in kubeflow/trainer#3227:
        "The env var is the path to the file, not the inline PEM."
        See also: kubeflow/sdk#368 for the reference implementation.

        If the cert file is missing or malformed, logs a warning and returns
        ``None`` (falls back to system CA bundle).  Training must not crash
        because of a controller-side certificate issue.
        """
        ca_cert_path = os.environ.get(ENV_STATUS_CA_CERT, "")
        if not ca_cert_path:
            return None

        ctx = ssl.create_default_context()
        try:
            ctx.load_verify_locations(ca_cert_path)
            return ctx
        except Exception:
            logger.warning(
                "Failed to load Kubeflow CA certificate from %s — falling "
                "back to system CA bundle.  Progress reporting may fail if "
                "the controller uses a self-signed cert.",
                ca_cert_path,
                exc_info=True,
            )
            return None


# ---------------------------------------------------------------------------
# KubeflowTrainerCallback — HuggingFace Trainer integration
# ---------------------------------------------------------------------------

# Conditional import: inherit from TrainerCallback when transformers is
# available (i.e. inside the training container where TRL/Unsloth run),
# fall back to object for SDK-only installations.  HuggingFace's
# CallbackHandler performs isinstance checks, so duck-typing alone is
# insufficient — the callback would be silently ignored.
try:
    from transformers import TrainerCallback as _TrainerCallback
except ImportError:
    _TrainerCallback = object  # type: ignore[misc,assignment]


class KubeflowTrainerCallback(_TrainerCallback):
    """HuggingFace Transformers ``TrainerCallback`` for Kubeflow progress.

    Automatically reports training progress and metrics to the Kubeflow
    Trainer status server on each logging step.

    Works with any HF Trainer-based framework:

    * HuggingFace Transformers ``Trainer``
    * TRL ``SFTTrainer``, ``DPOTrainer``, ``PPOTrainer``, ``ORPOTrainer``
    * Unsloth (patches TRL trainers transparently)

    Usage::

        from transformers import Trainer
        from kubeflow_llm_trainer.progress import KubeflowTrainerCallback

        trainer = Trainer(model=model, args=args, callbacks=[
            KubeflowTrainerCallback(),
        ])

    The callback is a no-op when running outside a Kubeflow TrainJob.

    Inherits from ``transformers.TrainerCallback`` when the ``transformers``
    package is available (inside training containers).  Falls back to
    ``object`` in SDK-only installations to avoid a hard dependency.
    """

    def __init__(self) -> None:
        self._reporter = KubeflowProgressReporter()
        self._start_time: float | None = None

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        """Called at the start of training."""
        self._start_time = time.time()
        if self._reporter.enabled:
            self._reporter.report(TrainerStatus(progress=0))

    def on_log(self, args: Any, state: Any, control: Any, logs: dict | None = None, **kwargs: Any) -> None:
        """Called on each logging step — reports progress and metrics."""
        if not self._reporter.enabled or state is None:
            return

        # Calculate progress percentage from step / max_steps.
        max_steps = getattr(state, "max_steps", 0)
        current_step = getattr(state, "global_step", 0)
        progress = int((current_step / max_steps) * 100) if max_steps > 0 else 0
        progress = min(progress, 100)

        # Estimate time remaining.
        eta: str | None = None
        if self._start_time and current_step > 0 and max_steps > 0:
            elapsed = time.time() - self._start_time
            steps_remaining = max_steps - current_step
            secs_per_step = elapsed / current_step
            eta_secs = int(steps_remaining * secs_per_step)
            if eta_secs >= 3600:
                eta = f"{eta_secs // 3600}h {(eta_secs % 3600) // 60}m"
            elif eta_secs >= 60:
                eta = f"{eta_secs // 60}m {eta_secs % 60}s"
            else:
                eta = f"{eta_secs}s"

        # Extract metrics from HF Trainer's log dict.
        metrics: list[Metric] = []
        if logs:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    metrics.append(Metric(name=key, value=f"{value:.6g}"))

        self._reporter.report(TrainerStatus(
            progress=progress,
            eta=eta,
            metrics=metrics,
        ))

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        """Called at the end of training — reports 100% completion with final metrics."""
        if self._reporter.enabled:
            # Include final metrics from the trainer's log_history if available.
            metrics: list[Metric] = []
            if state is not None:
                log_history = getattr(state, "log_history", None)
                if log_history and len(log_history) > 0:
                    last_log = log_history[-1]
                    for key, value in last_log.items():
                        if isinstance(value, (int, float)) and key != "epoch":
                            metrics.append(Metric(name=key, value=f"{value:.6g}"))
            self._reporter.report(TrainerStatus(progress=100, metrics=metrics))

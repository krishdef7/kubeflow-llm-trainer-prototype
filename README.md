# Dynamic LLM Trainer Framework for Kubeflow

**GSoC 2026 Prototype** — [KEP-2839](https://github.com/kubeflow/trainer/issues/2839) · Contributor: [@krishdef7](https://github.com/krishdef7)

---

## Problem

Kubeflow Trainer's SDK currently couples LLM fine-tuning to a single backend through a hardcoded dispatch in `BuiltinTrainer`:

```python
# Current SDK — hardcoded isinstance check
if isinstance(trainer.config, TorchTuneConfig):
    # ... TorchTune-specific logic
```

TorchTune is [no longer actively adding new features](https://github.com/kubeflow/trainer/issues/2839), which means Kubeflow users cannot access emerging post-training methods (DPO, PPO, ORPO, KTO) or faster backends (Unsloth, TRL) without modifying the SDK source code.

## Solution

This prototype implements a **pluggable backend architecture** that decouples the SDK from any single fine-tuning framework. Users switch backends by changing one field:

```python
# TorchTune (existing behavior, fully backward compatible)
LLMConfig(model="llama-3.2-1B", dataset="alpaca", backend_name="torchtune", ...)

# TRL — same model, same dataset, different engine
LLMConfig(model="llama-3.2-1B", dataset="alpaca", backend_name="trl", ...)

# Unsloth — ~2× faster, ~70% less memory
LLMConfig(model="llama-3.2-1B", dataset="alpaca", backend_name="unsloth", ...)
```

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    User Code                              │
│                                                          │
│  trainer = LLMTrainer(                                   │
│      config=LLMConfig(backend_name="trl", method="dpo"), │
│      resources_per_node={"gpu": 2},                      │
│  )                                                       │
│  client.train(name="my-job", trainer=trainer)             │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│              LLMTrainer.resolve()                         │
│                                                          │
│  1. Look up backend from BackendRegistry                 │
│  2. backend.validate(config)                             │
│  3. backend.to_container_spec(config) → ContainerSpec    │
│  4. Return ResolvedLLMTrainer                            │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│              BackendRegistry                              │
│                                                          │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐           │
│  │ TorchTune  │ │    TRL     │ │  Unsloth   │  ...      │
│  │  Backend   │ │  Backend   │ │  Backend   │           │
│  └─────┬──────┘ └─────┬──────┘ └─────┬──────┘           │
│        │              │              │                   │
│        └──────────────┼──────────────┘                   │
│                       │                                  │
│              LLMBackend (ABC)                            │
│         name | validate | to_container_spec              │
└──────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│         Kubernetes (unchanged controller)                 │
│                                                          │
│  TrainJob.spec.trainer.image   ← container_spec.image    │
│  TrainJob.spec.trainer.command ← container_spec.command   │
│  TrainJob.spec.trainer.args    ← container_spec.args     │
│  TrainJob.spec.runtimeRef      ← auto-discovered by      │
│                                  framework label          │
└──────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Config-driven vs. function-driven trainers

This framework is **config-driven**: users specify *what* to train (model, dataset, method, hyperparameters) and the backend decides *how* (entrypoint, image, CLI args).

This complements KEP-285's **function-driven** specialized trainers (`TorchTrainer`, `MPITrainer`), where users provide their own training code. The two coexist:

| | KEP-285 Specialized Trainers | KEP-2839 Dynamic LLM Trainer |
|---|---|---|
| **Pattern** | Function-driven | Config-driven |
| **User provides** | Training function | Model + dataset + hyperparams |
| **Use case** | "Run *my code* on N nodes" | "Fine-tune *this model* with *these params*" |
| **SDK type** | `TorchTrainer`, `MPITrainer` | `LLMTrainer` |

This distinction directly addresses [the question raised by @tariq-hasan](https://github.com/kubeflow/sdk/pull/308#discussion) on PR #308 about where config-driven trainers fit in the `BaseTrainer` hierarchy.

### 2. Zero controller changes for new backends

A new backend only needs to produce a valid `ContainerSpec`. The existing torch plugin in `pkg/runtime/framework/plugins/torch/` handles the rest. This means adding TRL, Unsloth, or LlamaFactory requires **zero Go code changes**.

### 3. Three registration paths

Backends are registered via (in order of precedence):

1. **Explicit registration:** `BackendRegistry.register(MyBackend())`
2. **Class decorator:** `@BackendRegistry.register`
3. **Entry-point discovery:** `[project.entry-points."kubeflow.llm_backends"]`

Path 3 enables third-party packages to ship backends:

```toml
# Third-party package's pyproject.toml
[project.entry-points."kubeflow.llm_backends"]
my_backend = "my_package:MyBackend"
```

### 4. Backward compatibility via transparent adaptation

Existing `BuiltinTrainer` code keeps working. In `TrainerClient.train()`:

```python
if isinstance(trainer, BuiltinTrainer):
    trainer = adapt_builtin_trainer(trainer)  # → LLMTrainer

if isinstance(trainer, LLMTrainer):
    resolved = trainer.resolve()
    # build TrainJob spec from resolved.container_spec
```

## Backend Capabilities Matrix

| Method | TorchTune | TRL | Unsloth |
|--------|:---------:|:---:|:-------:|
| SFT    | ✅ | ✅ | ✅ |
| LoRA   | ✅ | ✅* | ✅ |
| QLoRA  | ✅ | ✅* | ✅ |
| DPO    | ✅ | ✅ | ✅ |
| PPO    | ❌ | ✅ | ❌ |
| ORPO   | ❌ | ✅ | ✅ |
| KTO    | ❌ | ✅ | ❌ |
| Multi-node | ✅ | ✅ | ❌ |

\* TRL applies LoRA/QLoRA as a modifier on top of SFT/DPO/etc, not as a separate method.

## Project Structure

```
src/kubeflow_llm_trainer/
├── interface.py          # LLMBackend ABC, LLMConfig, ContainerSpec, enums
├── registry.py           # BackendRegistry with 3 registration paths
├── trainer.py            # LLMTrainer → ResolvedLLMTrainer
├── integration.py        # TrainJob spec generation (TrainerClient integration)
├── progress.py           # KEP-2779: Status server client + HF TrainerCallback
├── _compat.py            # BuiltinTrainer → LLMTrainer adapter
├── backends/
│   ├── torchtune.py      # TorchTune backend (backward compat)
│   ├── trl.py            # TRL backend (SFT/DPO/PPO/ORPO/KTO)
│   └── unsloth.py        # Unsloth backend (~2× faster)
└── entrypoints/
    ├── trl_runner.py     # Container entrypoint for TRL training
    └── unsloth_runner.py # Container entrypoint for Unsloth training

manifests/base/runtimes/
├── trl_distributed.yaml       # ClusterTrainingRuntime for TRL
└── unsloth_single_device.yaml # ClusterTrainingRuntime for Unsloth

tests/                    # 119 tests covering all paths
examples/                 # 4 runnable examples
```

## Running

```bash
# Install
pip install -e ".[dev]"

# Run tests (114 passing)
pytest tests/ -v

# Run examples
python examples/01_trl_sft.py
python examples/02_cross_backend_switching.py
python examples/03_custom_backend.py
python examples/04_migration_from_builtin.py
```

## Integration with Existing SDK

The changes required in the existing codebase are minimal (**~40 lines of diff**):

**`kubeflow/sdk/kubeflow/trainer/trainer_client.py`:**

```diff
+ from kubeflow_llm_trainer import LLMTrainer
+ from kubeflow_llm_trainer._compat import adapt_builtin_trainer

  def train(
      self,
      trainer: Optional[Union[
          "CustomTrainer",
          "BuiltinTrainer",
+         "LLMTrainer",
      ]] = None,
      ...
  ):
+     if isinstance(trainer, BuiltinTrainer):
+         trainer = adapt_builtin_trainer(trainer)
+
+     if isinstance(trainer, LLMTrainer):
+         resolved = trainer.resolve()
+         return self._submit_llm_trainjob(name, runtime, resolved, ...)
+
      # existing CustomTrainer path unchanged
```

**`kubeflow/trainer/pkg/runtime/...`:** No changes needed.

## GSoC Implementation Plan

### Phase 1: Core Framework (Weeks 1-4)
- Upstream the `LLMBackend` interface and `BackendRegistry`
- Refactor `BuiltinTrainer` to use `TorchTuneBackend` internally
- Integration tests against real ClusterTrainingRuntime

### Phase 2: TRL Backend (Weeks 5-8)
- Implement `TRLBackend` with SFT, DPO, PPO, ORPO, KTO
- Create `ClusterTrainingRuntime` manifests for TRL
- Build `trl-trainer` container image
- E2E tests on Kubernetes

### Phase 3: Unsloth + External Registration (Weeks 9-11)
- Implement `UnslothBackend` with optimized single-GPU training
- Entry-point based external backend discovery
- Documentation and migration guide

### Phase 4: Polish + LlamaFactory Exploration (Week 12)
- LlamaFactory backend proof-of-concept
- Performance benchmarks (TorchTune vs TRL vs Unsloth)
- Final documentation and blog post

## Related Work

- **KEP-2839:** [Kubeflow Dynamic LLM Trainer Framework](https://github.com/kubeflow/trainer/issues/2839) (tracking issue)
- **KEP-285:** [Specialized Trainers](https://github.com/kubeflow/sdk/pull/308) (complementary, function-driven)
- **KEP-2401:** [Kubeflow LLM Trainer V2](https://github.com/kubeflow/trainer/issues/2321) (original TorchTune integration)
- **PR #3227:** [TrainJob progress tracking](https://github.com/kubeflow/trainer/pull/3227) — this prototype includes SDK-side integration
- **PR #308 discussion:** [Config-driven vs function-driven trainers](https://github.com/kubeflow/sdk/pull/308) — directly addressed by this prototype

## Progress Reporting Integration (KEP-2779)

This prototype includes SDK-side integration for the TrainJob progress tracking
feature being implemented in [kubeflow/trainer#3227](https://github.com/kubeflow/trainer/pull/3227)
by [@robert-bell](https://github.com/robert-bell).

The `progress.py` module provides:

1. **`KubeflowProgressReporter`** — a standalone HTTP client that POSTs
   progress updates (progress %, ETA, custom metrics) to the Kubeflow Trainer
   status server.

2. **`KubeflowTrainerCallback`** — a HuggingFace Transformers `TrainerCallback`
   that automatically reports progress on each logging step.  Works with TRL's
   `SFTTrainer`, `DPOTrainer`, `PPOTrainer`, and Unsloth (which patches TRL).

The entrypoints (`trl_runner.py`, `unsloth_runner.py`) automatically inject the
callback when the status server env vars are detected — zero user configuration.

```python
# The callback is also usable standalone:
from kubeflow_llm_trainer.progress import KubeflowTrainerCallback

trainer = SFTTrainer(
    model=model,
    args=training_args,
    callbacks=[KubeflowTrainerCallback()],  # auto-reports to Kubeflow
)
```

## License

Apache License 2.0

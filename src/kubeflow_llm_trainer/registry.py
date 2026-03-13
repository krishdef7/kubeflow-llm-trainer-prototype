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

"""Backend registry for dynamic LLM trainer framework.

The registry supports three registration paths — listed in order of
precedence:

1. **Explicit registration** via ``BackendRegistry.register(backend)``.
2. **Decorator registration** via ``@BackendRegistry.register``.
3. **Entry-point discovery** via the ``kubeflow.llm_backends`` group.

Path (3) is what makes the system truly pluggable: a third-party package
can ship an LLM backend by adding an entry point in its ``pyproject.toml``:

.. code-block:: toml

   [project.entry-points."kubeflow.llm_backends"]
   my_backend = "my_package.backend:MyBackend"

The registry lazily discovers entry points the first time ``get()`` is called
for an unknown backend name.

Thread-safety
-------------
The registry uses a module-level dict.  In the Kubeflow SDK the registry is
accessed from a single thread (the main thread that calls
``TrainerClient.train()``), so locking is unnecessary.  If the SDK ever
becomes async-capable this should be revisited.
"""

from __future__ import annotations

import importlib.metadata
import logging
from typing import Type

from kubeflow_llm_trainer.interface import LLMBackend

logger = logging.getLogger(__name__)

_ENTRY_POINT_GROUP = "kubeflow.llm_backends"


class BackendRegistry:
    """Singleton registry mapping backend names to ``LLMBackend`` instances.

    Usage::

        # Explicit registration
        BackendRegistry.register(TorchTuneBackend())

        # Decorator registration
        @BackendRegistry.register
        class TRLBackend(LLMBackend):
            ...

        # Retrieval
        backend = BackendRegistry.get("trl")

        # Listing
        for name in BackendRegistry.list_backends():
            print(name)
    """

    _backends: dict[str, LLMBackend] = {}
    _entry_points_loaded: bool = False

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, backend: LLMBackend | Type[LLMBackend]) -> LLMBackend | Type[LLMBackend]:
        """Register a backend instance or class.

        Can be used as a decorator or called directly::

            # As decorator on a class
            @BackendRegistry.register
            class MyBackend(LLMBackend):
                ...

            # Direct registration of an instance
            BackendRegistry.register(MyBackend())

        Args:
            backend: An ``LLMBackend`` *instance* or *class*.  If a class is
                passed it is instantiated with no arguments.

        Returns:
            The original input (for decorator compatibility).

        Raises:
            TypeError: If *backend* is not an ``LLMBackend`` subclass/instance.
        """
        if isinstance(backend, type):
            # Decorator on a class — instantiate it.
            if not issubclass(backend, LLMBackend):
                raise TypeError(
                    f"Expected an LLMBackend subclass, got {backend!r}"
                )
            instance = backend()
            cls._backends[instance.name] = instance
            logger.debug("Registered LLM backend %r (class)", instance.name)
            return backend  # return the class unchanged (decorator pattern)
        elif isinstance(backend, LLMBackend):
            cls._backends[backend.name] = backend
            logger.debug("Registered LLM backend %r (instance)", backend.name)
            return backend
        else:
            raise TypeError(
                f"Expected an LLMBackend instance or class, got {type(backend)!r}"
            )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    @classmethod
    def get(cls, name: str) -> LLMBackend:
        """Retrieve a backend by name.

        If the name is not in the explicit registry, entry points are lazily
        loaded before giving up.

        Args:
            name: Backend identifier (e.g. ``"torchtune"``).

        Returns:
            The matching ``LLMBackend`` instance.

        Raises:
            KeyError: If no backend with *name* is registered.
        """
        if name in cls._backends:
            return cls._backends[name]

        # Lazy entry-point discovery.
        if not cls._entry_points_loaded:
            cls._load_entry_points()
            if name in cls._backends:
                return cls._backends[name]

        available = ", ".join(sorted(cls._backends)) or "(none)"
        raise KeyError(
            f"No LLM backend registered with name {name!r}. "
            f"Available backends: {available}. "
            f"Install a package that provides the {_ENTRY_POINT_GROUP!r} "
            f"entry point, or register a backend explicitly with "
            f"BackendRegistry.register()."
        )

    @classmethod
    def get_default(cls) -> LLMBackend:
        """Return the default backend.

        Resolution order:
        1. ``"torchtune"`` if registered (backward-compatible default).
        2. The first registered backend (alphabetical).
        3. Raise ``RuntimeError`` if the registry is empty.
        """
        cls._ensure_loaded()
        if "torchtune" in cls._backends:
            return cls._backends["torchtune"]
        if cls._backends:
            first_name = sorted(cls._backends)[0]
            return cls._backends[first_name]
        raise RuntimeError(
            "No LLM backends registered.  Install at least one backend "
            "package (e.g. kubeflow[torchtune]) or register a backend "
            "explicitly."
        )

    # ------------------------------------------------------------------
    # Listing / inspection
    # ------------------------------------------------------------------

    @classmethod
    def list_backends(cls) -> list[str]:
        """Return sorted list of registered backend names."""
        cls._ensure_loaded()
        return sorted(cls._backends)

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check whether a backend is registered (including entry points)."""
        cls._ensure_loaded()
        return name in cls._backends

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _ensure_loaded(cls) -> None:
        if not cls._entry_points_loaded:
            cls._load_entry_points()

    @classmethod
    def _load_entry_points(cls) -> None:
        """Discover and load backends from ``kubeflow.llm_backends`` entry points."""
        cls._entry_points_loaded = True
        try:
            eps = importlib.metadata.entry_points()
            # Python 3.12+ returns a SelectableGroups; 3.10-3.11 returns a dict.
            if hasattr(eps, "select"):
                group = eps.select(group=_ENTRY_POINT_GROUP)
            else:
                group = eps.get(_ENTRY_POINT_GROUP, [])

            for ep in group:
                if ep.name in cls._backends:
                    continue  # explicit registration takes precedence
                try:
                    backend_cls = ep.load()
                    instance = backend_cls()
                    cls._backends[instance.name] = instance
                    logger.debug(
                        "Loaded LLM backend %r from entry point %s",
                        instance.name,
                        ep.value,
                    )
                except Exception:
                    logger.warning(
                        "Failed to load LLM backend from entry point %s",
                        ep.value,
                        exc_info=True,
                    )
        except Exception:
            logger.warning(
                "Entry-point discovery failed for group %r",
                _ENTRY_POINT_GROUP,
                exc_info=True,
            )

    @classmethod
    def _reset(cls) -> None:
        """Clear all registered backends. **For testing only.**"""
        cls._backends.clear()
        # Mark as loaded to prevent entry-point re-discovery in tests.
        cls._entry_points_loaded = True

"""Python entry point for the AutoDraft target server.

Symmetric counterpart to ``autodraft.local_runner.run_local`` but for the
target side. Today users start the target server with ``./run_target.sh``
which shells out to ``python -m evaluation.eval_autodraft_target``;
``serve_target`` lets them do the same from a ``.py`` file.

Always runs in **lazy load** mode: the server starts up without any model
on the GPU and waits for the draft side to issue a ``reload_model`` RPC
specifying the model and quantization it wants. This means the user does
**not** need to repeat the model name / quantization on the target side —
draft is the source of truth.

Mirrors ``run_local``'s conventions:
- import-light at module load (no torch / no evaluation runtime)
- HF token resolved + exported to env BEFORE the lazy runtime import
- raises :class:`LocalRuntimeUnavailableError` if the source checkout / heavy
  deps are missing
- forwards a curated overlay; unknown kwargs are passed through to the inner
  ``serve()``

This function **blocks forever** (server loop). Run it in its own process or
its own thread.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Optional

from .errors import LocalRuntimeUnavailableError


def _import_runtime():
    try:
        from evaluation.eval_autodraft_target import serve, set_seed  # type: ignore
    except ImportError as exc:
        raise LocalRuntimeUnavailableError(
            "autodraft.target_server could not import the AutoDraft target "
            "runtime (evaluation/eval_autodraft_target.py). The wheel ships "
            "the runtime, so this usually means the heavy dependencies "
            "(torch, transformers, accelerate) are not installed in this "
            "environment. Install with `pip install autodraft-sd` (or "
            "`pip install -e .` from a source checkout) and make sure your "
            "PyTorch wheel matches your CUDA version."
        ) from exc
    return serve, set_seed


def _apply_hf_token(hf_token: Optional[str]) -> None:
    """Same protocol as ``autodraft.local_runner._apply_hf_token`` — kept
    duplicated rather than imported so this module stays import-light."""
    effective = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if effective:
        os.environ["HF_TOKEN"] = effective
        os.environ["HUGGING_FACE_HUB_TOKEN"] = effective


def _ensure_data_dir() -> None:
    """Same protocol as ``autodraft.local_runner._ensure_data_dir``: default
    to ``./data`` (CWD-relative) so caches land where the user ran the
    script. Override via ``AUTODRAFT_DATA_DIR``."""
    if not os.environ.get("AUTODRAFT_DATA_DIR"):
        os.environ["AUTODRAFT_DATA_DIR"] = str(Path.cwd() / "data")
    root = Path(os.environ["AUTODRAFT_DATA_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "profile").mkdir(parents=True, exist_ok=True)


def serve_target(
    host: str = "0.0.0.0",
    port: int = 26001,
    server_name: str = "autodraft",
    hf_token: Optional[str] = None,
    enable_auto_target_profile: bool = True,
    enable_auto_server_draft_profile: bool = True,
    enable_gpu_monitor: bool = False,
    gpu_monitor_interval: float = 0.05,
    fix_gpu_clock: bool = False,
    graphics_clock: Optional[int] = None,
    memory_clock: Optional[int] = None,
    seed: Optional[int] = None,
    deterministic: bool = False,
    debug: bool = False,
    output_file: Optional[str] = None,
    int8_cpu_offload: bool = False,
    **kwargs: Any,
) -> None:
    """Start the AutoDraft target server in lazy-load mode. **Blocks forever.**

    The server starts listening immediately with **no model loaded**. When
    the draft side connects, it sends a ``reload_model`` RPC carrying the
    target model id, quantization, and device_map; the target loads
    whatever the draft asks for. That means there is no ``target_model`` /
    ``quantization`` / ``device_map`` parameter here — the draft owns those
    decisions and you only need to set them on the ``Autodraft`` side.

    Parameters
    ----------
    host : str, optional
        Bind address. Default ``"0.0.0.0"`` (listen on all interfaces).
    port : int, optional
        TCP port to listen on. Default ``26001``. Must match the
        ``target_port`` you pass to ``Autodraft(...)``.
    server_name : str, optional
        Profile / reference cache key. Must match the ``server_name`` the
        draft side uses (default ``"autodraft"`` on both ends). Different
        ``server_name`` ⇒ separate cache slot.
    hf_token : str, optional
        HF access token for gated repositories. ``None`` falls back to
        ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` env vars. Treat as secret.
    enable_auto_target_profile : bool, optional
        Allow draft-initiated automatic target profiling RPCs. Required for
        the first run on a new ``(server, model, quantization)`` combo.
        Default ``True``.
    enable_auto_server_draft_profile : bool, optional
        Allow draft-initiated automatic server-side draft profiling for
        Server-Only / autoregressive workloads. Default ``True``.
    enable_gpu_monitor, gpu_monitor_interval, fix_gpu_clock, graphics_clock, memory_clock :
        GPU monitoring / clock-pinning knobs. Defaults match ``run_target.sh``.
    seed, deterministic :
        Reproducibility knobs. ``seed=None`` (default) leaves random state
        unchanged.
    debug : bool, optional
        Enable verbose ``[target]`` logging. Default ``False``.
    output_file : str, optional
        Path to dump performance stats; default ``None`` (no stats file).
    int8_cpu_offload : bool, optional
        Forwarded to bitsandbytes when the draft requests an 8-bit reload.
        Default ``False``.
    **kwargs
        Extra fields injected onto the ``args`` Namespace passed into the
        underlying ``serve()``. Useful for forward-compat options the
        runtime adds.
    """

    if not isinstance(port, int) or isinstance(port, bool):
        raise ValueError("port must be int")

    # Set HF env vars BEFORE the lazy runtime import. huggingface_hub reads
    # them at import time, and the target runtime resolves
    # ``from_pretrained(token=...)`` from os.environ.
    _apply_hf_token(hf_token)
    _ensure_data_dir()

    serve, set_seed = _import_runtime()

    if seed is not None:
        set_seed(int(seed))

    # Lazy mode: no startup model, no quantization choice, no draft model
    # loaded on the target host. The draft will send all of that in its
    # ``reload_model`` RPC. We hand placeholder defaults to the underlying
    # ``serve()`` because some of its positional args are non-Optional.
    placeholder_quantization = "8bit"
    placeholder_device_map = "auto"

    # serve() reads many additional fields off the args Namespace via getattr
    # (server_name, enable_auto_target_profile, enable_auto_server_draft_profile,
    # base_model_path, temperature, quantization, load_in_8bit, load_in_4bit,
    # enable_gpu_monitor, gpu_monitor_interval, fix_gpu_clock, graphics_clock,
    # memory_clock, device_map, seed, deterministic, debug). Build a Namespace
    # that covers all of them.
    args_ns = argparse.Namespace(
        host=host,
        port=port,
        base_model_path="",  # lazy
        temperature=0.0,
        load_in_8bit=False,
        load_in_4bit=False,
        int8_cpu_offload=int8_cpu_offload,
        device_map=placeholder_device_map,
        enable_gpu_monitor=enable_gpu_monitor,
        gpu_monitor_interval=gpu_monitor_interval,
        output_file=output_file,
        fix_gpu_clock=fix_gpu_clock,
        graphics_clock=graphics_clock,
        memory_clock=memory_clock,
        debug=debug,
        seed=seed,
        deterministic=deterministic,
        draft_model_path=None,  # lazy
        eager_load=False,
        server_name=server_name,
        enable_auto_target_profile=enable_auto_target_profile,
        enable_auto_server_draft_profile=enable_auto_server_draft_profile,
        quantization=placeholder_quantization,
    )
    # Allow callers to inject additional Namespace fields via **kwargs without
    # forcing us to enumerate every future flag.
    for k, v in kwargs.items():
        setattr(args_ns, k, v)

    serve(
        host,
        port,
        "",  # base_model_path — lazy load fills this in via reload_model RPC
        0.0,  # temperature placeholder
        placeholder_quantization,
        int8_cpu_offload,
        placeholder_device_map,
        enable_gpu_monitor,
        gpu_monitor_interval,
        output_file,
        fix_gpu_clock,
        graphics_clock,
        memory_clock,
        args=args_ns,
        debug=debug,
        draft_model_path=None,
        preload_model_on_start=False,
    )

"""In-process wrapper around the AutoDraft draft runtime.

The chat UI (``chat_ui/autodraft_service.py``) drives the same flow by
subprocessing ``python -m evaluation.eval_autodraft_draft`` with a one-line
mt_bench JSONL question file and reading the resulting answer file. We follow
that pattern here, but call ``run_draft`` directly in-process instead of
spawning a subprocess.

Notes:
- ``evaluation`` and ``opt_classic`` are NOT shipped inside the ``autodraft``
  wheel. ``run_local`` therefore only works from a source checkout. Standalone
  PyPI installs raise :class:`LocalRuntimeUnavailableError`.
- ``run_draft`` opens a socket to ``target_host:target_port`` and a target
  server (started via ``./run_target.sh``) must be reachable there before
  this function can produce output. That is architectural; we do not hide it.
"""

from __future__ import annotations

import inspect
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from .errors import LocalRuntimeUnavailableError


def _import_runtime():
    """Lazy-import the heavy AutoDraft draft runtime."""
    try:
        from evaluation.eval_autodraft_draft import run_draft, set_seed  # type: ignore
    except ImportError as exc:
        raise LocalRuntimeUnavailableError(
            "autodraft.local_runner could not import the AutoDraft draft "
            "runtime (evaluation/eval_autodraft_draft.py). The wheel ships "
            "the runtime, so this usually means the heavy dependencies "
            "(torch, transformers, fastchat, shortuuid, tqdm, accelerate) "
            "are not installed in this environment. Install with "
            "`pip install autodraft-sd` (or `pip install -e .` from a "
            "source checkout) and make sure your PyTorch wheel matches "
            "your CUDA version. engine.run also needs a reachable target "
            "server (start one via examples/target.py or autodraft.serve_target)."
        ) from exc
    return run_draft, set_seed


# UI ↔ run_draft flag translation. The chat UI's algorithm dropdown
# (``AutoDraft`` / ``Server-Only`` / ``Server-Only-AR`` / ``OPT-Tree`` /
# ``Fixed-tree``) is a higher-level concept than ``run_draft``'s flags, so
# we translate it here the same way ``chat_ui/autodraft_service.py:_build_base_command``
# does.
def _algorithm_overlay(algorithm: str) -> Dict[str, Any]:
    algo = (algorithm or "AutoDraft").strip()
    overlay: Dict[str, Any] = {}
    if algo == "Server-Only":
        overlay.update(force_server_only=True)
    elif algo == "Server-Only-AR":
        overlay.update(
            force_server_only=True,
            nodes=1,
            max_depth=1,
            fixed_width=True,
            fixed_width_value=1,
            fixed_nnodes=True,
            fixed_depth=True,
        )
    elif algo == "OPT-Tree":
        overlay.update(opt_tree=True, disable_server_only=True)
    elif algo == "Fixed-tree":
        overlay.update(
            fixed_depth=True,
            fixed_nnodes=True,
            fixed_width=True,
            disable_server_only=True,
        )
    else:
        # AutoDraft (default)
        overlay.update(disable_server_only=True)
    return overlay


def _normalize_quantization(value: Optional[str]) -> str:
    return (value or "none").lower()


# ``cs`` (cost-sensitivity) preset table. The runtime accepts a number in
# [0, 1] where 0 = pure TPS focus and 1 = pure cost focus. Most users
# never want a fine-grained slider, so we expose three named presets and
# default to ``"balanced"``. Numeric values are still accepted for power
# users.
_CS_PRESETS = {
    "tps": 0.0,
    "balanced": 0.5,
    "cost": 1.0,
}


def _resolve_cost_sensitivity(cs: Any) -> float:
    if cs is None:
        return _CS_PRESETS["balanced"]
    if isinstance(cs, str):
        key = cs.strip().lower()
        if key not in _CS_PRESETS:
            raise ValueError(
                f"cs must be one of {sorted(_CS_PRESETS)} or a number in [0, 1]; "
                f"got {cs!r}"
            )
        return float(_CS_PRESETS[key])
    try:
        value = float(cs)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"cs must be one of {sorted(_CS_PRESETS)} or a number in [0, 1]; "
            f"got {cs!r}"
        ) from exc
    if value < 0.0 or value > 1.0:
        raise ValueError(f"cs numeric value must be in [0, 1]; got {value}")
    return value


# ``cost`` parameter mapping. ``run_draft``'s ``objective_metric`` accepts
# exactly these strings; we expose the same set so users don't have to look
# up the runtime constant.
_COST_METRICS = {
    "total_cost",
    "api_cost",
    "draft_energy",
    "target_energy",
}


def _resolve_cost_metric(cost: Optional[str]) -> str:
    metric = (cost or "total_cost").strip().lower()
    if metric not in _COST_METRICS:
        raise ValueError(
            f"cost must be one of {sorted(_COST_METRICS)}; got {cost!r}"
        )
    return metric


# Generation-stat fields we surface as a flat ``stats`` dict on the result.
# The runtime stores everything under ``integrated_result.generation_stats``
# (lots of keys); we hand-pick the small set the user asked for plus cost
# breakdowns. Missing keys fall through as ``None``.
_STAT_FIELDS = {
    "total_steps": "total_steps",
    "total_new_tokens": "total_new_tokens",
    "total_time_seconds": "total_time_seconds",
    "tokens_per_second": "tokens_per_second",
    "tokens_per_step": "tokens_per_step",
    "avg_tree_width": "avg_tree_width",
    "avg_tree_depth": "avg_tree_depth",
    "avg_nnodes": "avg_final_nodes",
    "avg_accept_length": "avg_accept_length",
    "acceptance_ratio_avg": "acceptance_ratio_avg",
    "total_cost": "total_cost",
    "api_cost": "api_cost",
    "draft_cost": "draft_cost",
    "target_cost": "target_cost",
}


def _extract_stats(raw_result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Pull the user-facing summary fields out of ``generation_stats``."""
    if not raw_result:
        return None
    gen = raw_result.get("generation_stats") or {}
    return {public: gen.get(internal) for public, internal in _STAT_FIELDS.items()}


def _norm_token(value: Any) -> str:
    """Mirror chat_ui/_norm_reference_token: lowercase, replace non-alnum
    with '-', collapse double dashes, fall back to 'none' on empty input.

    Used to keep trade-off filenames in sync with the reference cache
    naming convention so the two stay paired in ``data/`` listings.
    """
    s = str(value or "").split("/")[-1].lower()
    out = []
    for ch in s:
        out.append(ch if ("a" <= ch <= "z") or ("0" <= ch <= "9") else "-")
    v = "".join(out).strip("-")
    while "--" in v:
        v = v.replace("--", "-")
    return v or "none"


def _build_tradeoff_basename(
    *,
    server_name: str,
    target_model: str,
    device_name: str,
    draft_model: str,
    target_quantization: str,
    draft_quantization: str,
    bench_name: str,
    cost_metric: str,
    objective_mode: str,
) -> str:
    """Build a stable basename keyed on the conditions that affect the
    trade-off curve. Same conditions → same filename → repeated
    ``engine.run`` calls overwrite the same file rather than piling up
    timestamped duplicates."""
    return (
        f"tradeoff_{_norm_token(server_name)}_{_norm_token(target_model)}_"
        f"{_norm_token(device_name)}_{_norm_token(draft_model)}_"
        f"tq-{_norm_token(target_quantization)}_dq-{_norm_token(draft_quantization)}_"
        f"{_norm_token(bench_name)}_{_norm_token(cost_metric)}_"
        f"{_norm_token(objective_mode)}"
    )


def _save_tradeoff_artifacts(
    raw_result: Optional[Dict[str, Any]],
    dest_dir: Path,
    basename: str,
) -> Optional[Dict[str, Optional[str]]]:
    """Save the cs0~1 reference trade-off curve under a stable name.

    Always writes a JSON snapshot (cheap, no extra deps). Also tries to
    render a PNG via matplotlib (a base dep). Returns
    a dict with the paths of whatever got written, or ``None`` if the
    runtime didn't produce a curve (typically because the reference cache
    was already complete and no curve was rebuilt this run).

    The basename is conditions-hashed (see :func:`_build_tradeoff_basename`)
    so repeated calls with the same conditions overwrite the previous
    file. The on-disk artifact therefore always reflects the latest
    reference cache state for that condition set.
    """
    if not raw_result:
        return None
    exp = raw_result.get("experiment_info") or {}
    curve = exp.get("reference_tradeoff_curve_cs0_1")
    if not isinstance(curve, list) or not curve:
        return None

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    json_path = dest_dir / f"{basename}.json"
    json_path.write_text(json.dumps(curve, indent=2), encoding="utf-8")

    png_path: Optional[Path] = None
    try:
        import matplotlib  # type: ignore
        matplotlib.use("Agg")  # headless — no $DISPLAY needed
        import matplotlib.pyplot as plt  # type: ignore

        cs_values = [float(p.get("cost_sensitivity", 0.0)) for p in curve]
        tps_values = [float(p.get("predicted_tps", 0.0)) for p in curve]
        cost_values = [float(p.get("predicted_metric_per_1m_token", 0.0)) for p in curve]

        fig, ax_tps = plt.subplots(figsize=(9, 5))
        ax_tps.plot(cs_values, tps_values, color="tab:blue", marker="o", label="predicted TPS")
        ax_tps.set_xlabel("cost_sensitivity (0=TPS, 1=cost)")
        ax_tps.set_ylabel("predicted TPS", color="tab:blue")
        ax_tps.tick_params(axis="y", labelcolor="tab:blue")

        ax_cost = ax_tps.twinx()
        ax_cost.plot(cs_values, cost_values, color="tab:red", marker="s", label="predicted cost / 1M tokens")
        ax_cost.set_ylabel("predicted cost per 1M tokens", color="tab:red")
        ax_cost.tick_params(axis="y", labelcolor="tab:red")

        ax_tps.set_title(
            f"AutoDraft reference trade-off — {exp.get('draft_model','?')} → {exp.get('base_model','?')}"
        )
        fig.tight_layout()
        png_path = dest_dir / f"{basename}.png"
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
    except ImportError:
        # matplotlib should be a base dep, but be defensive in case the
        # user uninstalled it or installed the wheel with --no-deps.
        print(
            "[autodraft] matplotlib not installed; trade-off PNG skipped. "
            "Install with `pip install matplotlib` (or reinstall autodraft-sd) "
            "to enable PNG rendering."
        )
    except Exception as exc:  # rendering failed for some other reason
        print(f"[autodraft] tradeoff PNG render failed: {exc}")

    return {"json": str(json_path), "png": str(png_path) if png_path else None}


def _write_question_file(path: Path, input_text: str) -> None:
    """Serialize a single user prompt as one-line mt_bench JSONL."""
    record = {
        "question_id": 1,
        "category": "autodraft",
        "turns": [input_text],
    }
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")


def _parse_answer_file(path: Path) -> Optional[str]:
    """Read the last JSONL row produced by run_draft and return the final
    turn's text. Mirrors ``chat_ui/autodraft_service.py:_parse_answer_file``.

    Used as a fallback if the integrated results file (see
    ``_parse_integrated_result``) is not present.
    """
    if not path.exists():
        return None
    try:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            return None
        row = json.loads(lines[-1])
        choices = row.get("choices") or []
        if not choices:
            return None
        turns = choices[0].get("turns") or []
        if not turns:
            return None
        return str(turns[-1]).strip()
    except (OSError, ValueError, KeyError):
        return None


def _find_integrated_result_file(tmpdir: Path, answer_file: Path) -> Optional[Path]:
    """Locate the ``{basename}_results_{timestamp}.json`` file run_draft writes.

    ``run_draft`` ignores the literal ``answer_file`` path and instead writes
    its output to ``<answer_file_stem>_results_<timestamp>.json`` in the same
    directory (see eval_autodraft_draft.py around line 8665). We pick the
    most recent matching file and exclude the ``_trimmed`` companion.
    """
    stem = answer_file.stem  # e.g. "answer"
    candidates = sorted(
        (p for p in tmpdir.glob(f"{stem}_results_*.json") if not p.stem.endswith("_trimmed")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _parse_integrated_result(path: Path) -> Optional[Dict[str, Any]]:
    """Parse the top-level integrated result file run_draft produces.

    Returns a dict ``{generated_text, answer_row, raw}`` if the answer can
    be extracted; ``None`` otherwise. The integrated format is::

        {
          ...,
          "answers": [
            {"question_id": ..., "choices": [{"turns": [...], ...}], ...}
          ]
        }
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    answers = raw.get("answers") or []
    if not answers:
        return None
    answer_row = answers[-1]
    choices = answer_row.get("choices") or []
    if not choices:
        return None
    turns = choices[0].get("turns") or []
    if not turns:
        return None
    return {
        "generated_text": str(turns[-1]).strip(),
        "answer_row": answer_row,
        "raw": raw,
    }


def _apply_hf_token(hf_token: Optional[str]) -> None:
    """Export the user-supplied HF access token via env vars.

    The AutoDraft draft runtime calls ``from_pretrained`` with
    ``token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")``,
    so setting either of those env vars before the runtime is imported is
    sufficient. We set both for compatibility (the official name is
    ``HF_TOKEN``; ``HUGGING_FACE_HUB_TOKEN`` is the legacy alias still read
    by huggingface_hub).
    """
    effective = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if effective:
        os.environ["HF_TOKEN"] = effective
        os.environ["HUGGING_FACE_HUB_TOKEN"] = effective


def _resolve_device_name() -> str:
    """Pick a stable identifier for the local GPU, mirroring
    ``chat_ui/autodraft_service.py:_resolve_device_name``.

    Why this matters: profile cache filenames embed the device name, so
    different GPUs get different cache slots (a Llama-2-7B latency profile
    for an RTX 5080 isn't reusable on a Tesla V100). When ``device_name``
    is falsy, the runtime's profile-load path early-returns and **no
    auto-profiling happens**, which is exactly the symptom we want to fix
    here.

    Resolution order:
    1. ``AUTODRAFT_DEVICE_NAME`` env var (lets users pin their own label)
    2. ``torch.cuda.get_device_name(0)`` (auto-detect)
    3. ``"unknown-gpu"`` (always non-empty so the profile path runs)
    """
    configured = os.environ.get("AUTODRAFT_DEVICE_NAME", "").strip()
    if configured:
        return configured
    try:
        import torch  # lazy: keep autodraft import-light
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            name = str(torch.cuda.get_device_name(0)).strip()
            if name:
                return name
    except Exception:
        pass
    return "unknown-gpu"


def _ensure_data_dir() -> None:
    """Pick a writable location for the runtime's profile/reference caches.

    The runtime resolves ``data/profile`` and ``data/reference`` relative
    to its own ``parent_dir``. After ``pip install autodraft-sd`` that
    resolves to ``site-packages/data`` which is awkward (sometimes
    read-only, pollutes the install tree). Default to ``./data`` so the
    cache lives next to wherever the user launched their script — that
    matches the source-checkout behavior (where users run from the repo
    root and the runtime writes to ``<repo>/data``) and gives PyPI users a
    visible, project-local cache. Honor ``AUTODRAFT_DATA_DIR`` if the user
    wants a fixed location.
    """
    if not os.environ.get("AUTODRAFT_DATA_DIR"):
        os.environ["AUTODRAFT_DATA_DIR"] = str(Path.cwd() / "data")
    root = Path(os.environ["AUTODRAFT_DATA_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    for sub in ("profile", "reference"):
        (root / sub).mkdir(parents=True, exist_ok=True)


def run_local(
    draft_model: str,
    target_model: str,
    draft_quantization: Optional[str],
    target_quantization: Optional[str],
    input_text: str,
    proactive: bool = False,
    cs: Any = "balanced",
    cost: str = "total_cost",
    target_host: str = "127.0.0.1",
    target_port: int = 26001,
    hf_token: Optional[str] = None,
    save_tradeoff: bool = True,
    tradeoff_dir: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run a single-prompt AutoDraft decode and return the generated text.

    Internally this writes ``input_text`` to a temporary one-line mt_bench
    JSONL file, calls ``evaluation.eval_autodraft_draft.run_draft``
    in-process (the same function the chat UI subprocesses), and parses the
    answer file the script writes. ``run_draft`` requires a target server to
    be reachable at ``target_host:target_port``.

    Parameters
    ----------
    cs:
        Cost-sensitivity preset (``"tps"`` / ``"balanced"`` / ``"cost"``) or
        a number in ``[0, 1]``. Default ``"balanced"`` (== 0.5).
    cost:
        Which cost objective to optimize. One of ``"total_cost"`` (default),
        ``"api_cost"``, ``"draft_energy"``, ``"target_energy"``.
    save_tradeoff:
        When True (default), save the reference trade-off curve as
        ``tradeoff_<timestamp>.json`` and (if matplotlib is installed) PNG.
    tradeoff_dir:
        Directory for trade-off artifacts. Defaults to
        ``$AUTODRAFT_DATA_DIR/tradeoff``.
    algorithm:
        Selects AutoDraft variant (``"AutoDraft"`` (default), ``"Server-Only"``,
        ``"Server-Only-AR"``, ``"OPT-Tree"``, ``"Fixed-tree"``). Pass via
        ``**kwargs``.

    Returns
    -------
    dict
        ``{generated_text, input_text, proactive, cs, cost, algorithm,
        stats, tradeoff_files, answer_row, raw_result}``. ``stats`` is a
        flat dict with tree width/depth/nnodes, accept length, and cost
        breakdowns; see ``_STAT_FIELDS``.
    """

    if not isinstance(input_text, str) or not input_text:
        raise ValueError("input_text must be a non-empty string")

    cs_value = _resolve_cost_sensitivity(cs)
    cost_metric = _resolve_cost_metric(cost)

    # Set HF env vars BEFORE the lazy runtime import. huggingface_hub reads
    # these at import time, and the draft runtime resolves its
    # ``from_pretrained(token=...)`` argument from os.environ at call time.
    _apply_hf_token(hf_token)
    _ensure_data_dir()

    run_draft, set_seed = _import_runtime()

    algorithm = kwargs.pop("algorithm", "AutoDraft")
    seed = kwargs.pop("seed", 4)
    set_seed(int(seed))

    # Curated overlay: only the values where the chat UI intentionally
    # diverges from run_draft's own defaults. Everything else is left to
    # run_draft to default.
    overlay: Dict[str, Any] = dict(
        host=target_host,
        port=target_port,
        base_model_path=target_model,
        draft_model_path=draft_model,
        bench_name="mt_bench",
        limit=1,
        num_choices=1,
        temperature=0.0,
        nodes=150,
        max_depth=15,
        device_map=os.getenv("AUTODRAFT_DEVICE_MAP", "cuda:0"),
        # device_name is required: when it's falsy, the runtime's profile-load
        # path early-returns and auto-profiling never runs.
        device_name=_resolve_device_name(),
        load_in_4bit=(_normalize_quantization(draft_quantization) == "4bit"),
        load_in_8bit=(_normalize_quantization(draft_quantization) == "8bit"),
        target_quantization=_normalize_quantization(target_quantization),
        draft_per_hour_cost=0.152,
        target_per_hour_cost=1.208,
        user_communication_cost_per_gb=2.3333333333,
        accept_length_margin=0.05,
        objective_metric=cost_metric,
        objective_selection_mode="blend",
        constraint_target="metric",
        cost_sensitivity=cs_value,
        online_profile_update=True,
        online_profile_lr=0.05,
        enable_gpu_monitor=True,
    )
    overlay.update(_algorithm_overlay(algorithm))
    if proactive:
        overlay.update(
            proactive_drafting=True,
            adaptive_proactive_threshold=True,
            proactive_threshold=0.0,
        )
    overlay.update(kwargs)

    # ``run_draft`` is a fixed-arg function (no **kwargs sink today). Filter
    # the overlay to parameters it actually accepts so we never hit "got
    # unexpected keyword argument" if the wrapper drifts ahead of the
    # runtime. If a future ``run_draft`` adds a **kwargs sink, accept
    # everything.
    sig = inspect.signature(run_draft)
    has_var_keyword = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if has_var_keyword:
        filtered = dict(overlay)
        dropped: list = []
    else:
        accepted = {
            name for name, p in sig.parameters.items()
            if p.kind is not inspect.Parameter.VAR_POSITIONAL
        }
        filtered = {k: v for k, v in overlay.items() if k in accepted}
        dropped = sorted(set(overlay) - accepted)
    if dropped:
        print(f"[autodraft] dropping unknown run_draft kwargs: {dropped}")

    with tempfile.TemporaryDirectory(prefix="autodraft-") as tmpdir:
        tmp = Path(tmpdir)
        question_file = tmp / "question.jsonl"
        answer_file = tmp / "answer.jsonl"
        _write_question_file(question_file, input_text)
        filtered.setdefault("question_file", str(question_file))
        filtered.setdefault("answer_file", str(answer_file))

        run_draft(**filtered)

        # ``run_draft`` writes the actual answers to a sibling file named
        # ``{stem}_results_{timestamp}.json`` rather than to ``answer_file``
        # itself. Look for that integrated file first; fall back to the
        # JSONL parser in case a future run_draft starts writing to
        # answer_file directly.
        generated_text: Optional[str] = None
        answer_row: Optional[Dict[str, Any]] = None
        raw_result: Optional[Dict[str, Any]] = None

        integrated_path = _find_integrated_result_file(tmp, answer_file)
        if integrated_path is not None:
            parsed = _parse_integrated_result(integrated_path)
            if parsed is not None:
                generated_text = parsed["generated_text"]
                answer_row = parsed["answer_row"]
                raw_result = parsed["raw"]

        if generated_text is None:
            # Fallback: maybe a future run_draft writes JSONL to answer_file.
            generated_text = _parse_answer_file(answer_file)

    stats = _extract_stats(raw_result)

    tradeoff_files: Optional[Dict[str, Optional[str]]] = None
    if save_tradeoff:
        # Stable filename keyed on the conditions that affect the curve.
        # Same conditions across multiple ``engine.run`` calls overwrite
        # the same file, so the artifact on disk always reflects the
        # latest reference state for that condition set.
        basename = _build_tradeoff_basename(
            server_name=str(filtered.get("server_name", "autodraft")),
            target_model=target_model,
            device_name=str(filtered.get("device_name", "")),
            draft_model=draft_model,
            target_quantization=_normalize_quantization(target_quantization),
            draft_quantization=_normalize_quantization(draft_quantization),
            bench_name=str(filtered.get("bench_name", "mt_bench")),
            cost_metric=cost_metric,
            objective_mode=str(filtered.get("objective_selection_mode", "blend")),
        )
        target_dir = Path(tradeoff_dir) if tradeoff_dir else Path(os.environ["AUTODRAFT_DATA_DIR"]) / "tradeoff"
        try:
            tradeoff_files = _save_tradeoff_artifacts(raw_result, target_dir, basename)
        except OSError as exc:
            print(f"[autodraft] could not write trade-off artifacts to {target_dir}: {exc}")

    return {
        "generated_text": generated_text,
        "input_text": input_text,
        "proactive": bool(proactive),
        "cs": cs,
        "cost": cost_metric,
        "algorithm": algorithm,
        "stats": stats,
        "tradeoff_files": tradeoff_files,
        "answer_row": answer_row,
        "raw_result": raw_result,
    }

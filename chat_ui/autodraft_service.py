import asyncio
import json
import math
import os
import re
import socket
import time
import uuid
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional
import urllib.error
from urllib.parse import urlparse

try:
    from .probe_runner import ProbeRequest, ProbeRunner
    from .recommendation_engine import build_recommendations
    from .server_registry import ServerRegistry
    from opt_classic.utils import recv_json_with_size, send_json_with_size
except Exception:
    from probe_runner import ProbeRequest, ProbeRunner
    from recommendation_engine import build_recommendations
    from server_registry import ServerRegistry
    from opt_classic.utils import recv_json_with_size, send_json_with_size


LINE_TPS_RE = re.compile(
    r"cost per 1M token:\s*([0-9]*\.?[0-9]+)\s*\|\s*tokens per second:\s*([0-9]*\.?[0-9]+)",
    re.IGNORECASE,
)
LINE_ENERGY_RE = re.compile(
    r"draft energy total:\s*([0-9]*\.?[0-9]+)\s*kWh\s*\|\s*draft energy per 1M token:\s*([0-9]*\.?[0-9]+)",
    re.IGNORECASE,
)
DEFAULT_TARGET_PROFILE_NODE_LIST = "10,20,30,40,50,60,70,80,90,100,110,120,130,140,150"
DETAILED_TARGET_PROFILE_NODE_LIST = (
    "10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150"
)

TARGET_DRAFT_COMPATIBILITY = {
    "meta-llama/llama-3.3-70b-instruct": {
        "meta-llama/llama-3.2-3b-instruct",
        "meta-llama/llama-3.2-1b-instruct",
    },
    "qwen/qwen2.5-32b-instruct": {
        "qwen/qwen2.5-3b-instruct",
        "qwen/qwen2.5-1.5b-instruct",
    },
    "qwen/qwen2.5-14b-instruct": {
        "qwen/qwen2.5-1.5b-instruct",
        "qwen/qwen2.5-0.5b-instruct",
    },
    "qwen/qwen3-32b": {
        "qwen/qwen3-0.6b",
    },
    "qwen/qwen3-14b": {
        "qwen/qwen3-0.6b",
    },
}


def _canonical_model_key(model_id: str) -> str:
    raw = str(model_id or "").strip().lower().split(":", 1)[0]
    if not raw:
        return ""
    if "llama-3.3-70b-instruct" in raw:
        return "meta-llama/llama-3.3-70b-instruct"
    if "llama-3.2-3b-instruct" in raw:
        return "meta-llama/llama-3.2-3b-instruct"
    if "llama-3.2-1b-instruct" in raw:
        return "meta-llama/llama-3.2-1b-instruct"
    if "qwen2.5-32b-instruct" in raw:
        return "qwen/qwen2.5-32b-instruct"
    if "qwen2.5-14b-instruct" in raw:
        return "qwen/qwen2.5-14b-instruct"
    if "qwen2.5-3b-instruct" in raw:
        return "qwen/qwen2.5-3b-instruct"
    if "qwen2.5-1.5b-instruct" in raw:
        return "qwen/qwen2.5-1.5b-instruct"
    if "qwen2.5-0.5b-instruct" in raw:
        return "qwen/qwen2.5-0.5b-instruct"
    if "qwen3-32b" in raw:
        return "qwen/qwen3-32b"
    if "qwen3-14b" in raw:
        return "qwen/qwen3-14b"
    if "qwen3-0.6b" in raw:
        return "qwen/qwen3-0.6b"
    return raw


def _normalize_quantization_mode(value: str, default: str = "none") -> str:
    q = str(value or default).strip().lower()
    if q in {"none", "4bit", "8bit"}:
        return q
    return str(default).strip().lower()


def _build_target_quantization_fallback_chain(requested: str) -> List[str]:
    preferred = _normalize_quantization_mode(requested, default="none")
    order = ["none", "8bit", "4bit"]
    start_idx = order.index(preferred) if preferred in order else 0
    return order[start_idx:]


def _is_target_memory_related_reload_error(message: str) -> bool:
    text = str(message or "").strip().lower()
    if not text:
        return False
    hints = (
        "cuda out of memory",
        "out of memory",
        "insufficient memory",
        "modules are dispatched on the cpu",
        "dispatched on the cpu",
        "cpu or the disk",
        "offload",
    )
    return any(h in text for h in hints)


class AutoDraftService:
    """Thin adapter to run eval_autodraft_draft as a subprocess."""

    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root)
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._stdout_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._run_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._chat_proc: Optional[asyncio.subprocess.Process] = None
        self._chat_stdout_task: Optional[asyncio.Task] = None
        self._chat_stderr_task: Optional[asyncio.Task] = None
        self._chat_queue: asyncio.Queue = asyncio.Queue()
        self._chat_settings_key: Optional[str] = None
        self._chat_stdin_lock = asyncio.Lock()

        self.last_line = ""
        self.last_error = ""
        self.last_result_json = None
        self.running = False
        self.exit_code = None
        self.phase = "idle"
        self.phase_label = "Idle"
        self.chat_ready = False
        self.feasible_constraint_range_per_1m = None
        self.feasible_tps_range = None
        self.reference_tradeoff_curve_cs0_1 = None
        self.reference_tradeoff_curve_by_constraint = None
        self.reference_cs_anchor_curve = None
        self.reference_constraint_anchor_curve = None
        self.current_settings: Dict = {}
        self.stats = {
            "gpu_energy": 0.0,
            "draft_cost": 0.0,
            "target_cost": 0.0,
            "throughput": 0.0,
            "cost_per_1m": 0.0,
            "draft_energy_per_1m": 0.0,
            "running": False,
        }
        self.registry = ServerRegistry(self.repo_root / "chat_ui" / "tmp" / "servers_registry.json")
        self.probe_runner = ProbeRunner()
        self.probe_rows = []
        self.probe_curves = []
        self.probe_curve_mode = "cost_sensitivity"
        self.probe_status = {"running": False, "last_error": "", "updated_at": None}
        self.recommendation_summary = {
            "fastest": None,
            "best_efficiency": None,
            "pareto_optimal_ids": [],
        }
        self.metric_preference = "total_cost"

    def _reset_tradeoff_state(self):
        # Clear all trade-off/reference/probe artifacts after explicit stop.
        self.feasible_constraint_range_per_1m = None
        self.feasible_tps_range = None
        self.reference_tradeoff_curve_cs0_1 = None
        self.reference_tradeoff_curve_by_constraint = None
        self.reference_cs_anchor_curve = None
        self.reference_constraint_anchor_curve = None
        self.probe_rows = []
        self.probe_curves = []
        self.probe_status = {"running": False, "last_error": "", "updated_at": time.time()}
        self.recommendation_summary = {
            "fastest": None,
            "best_efficiency": None,
            "pareto_optimal_ids": [],
        }

    def _is_reference_phase(self) -> bool:
        return str(self.phase).startswith("reference")

    @staticmethod
    def _looks_like_progress_line(line: str) -> bool:
        l = (line or "").lower()
        return ("loading checkpoint shards" in l) or ("it/s" in l and "%" in l)

    @staticmethod
    def _looks_like_error_line(line: str) -> bool:
        l = (line or "").lower()
        if not l:
            return False
        # Transformers emits this advisory on stderr; it is not a runtime failure.
        if "generationmixin" in l:
            return False
        return any(tok in l for tok in ("error", "failed", "exception", "traceback", "connectionrefused"))

    @staticmethod
    def _is_connection_refused_error(message: str) -> bool:
        m = str(message or "").lower()
        return (
            "connectionrefusederror" in m
            or "errno 111" in m
            or "connection refused" in m
            or "actively refused" in m
        )
        self.tmp_dir = self.repo_root / "chat_ui" / "tmp"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _emit_runtime_log(channel: str, message: str):
        msg = str(message or "").strip()
        if not msg:
            return
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}][{channel}] {msg}", flush=True)

    def _settings_key(self, settings: Dict) -> str:
        resolved = self._resolve_runtime_target_config(settings)
        return json.dumps(
            {
                "algorithm": settings.get("algorithm"),
                "draft_model_path": settings.get("draft_model_path"),
                "objective_metric": self._objective_metric_from_settings(settings),
                "objective_selection_mode": settings.get("objective_selection_mode", "blend"),
                "constraint_target": settings.get("constraint_target", "metric"),
                "selected_server_id": settings.get("selected_server_id"),
                "selected_model_id": settings.get("selected_model_id"),
                "target_quantization": settings.get("target_quantization", "none"),
                "draft_quantization": settings.get("draft_quantization", "none"),
                "benchmark_dataset": settings.get("benchmark_dataset", "mt_bench"),
                "runtime_host": resolved.get("host"),
                "runtime_port": resolved.get("port"),
                "runtime_base_model": resolved.get("base_model"),
                "runtime_protocol": resolved.get("selected_protocol"),
            },
            sort_keys=True,
            ensure_ascii=True,
        )

    @staticmethod
    def _objective_metric_from_metric_preference(metric_preference: Optional[str]) -> str:
        pref = str(metric_preference or "total_cost").lower()
        if pref in {"total_cost", "api_cost", "draft_energy", "target_energy"}:
            return pref
        return "total_cost"

    def _objective_metric_from_settings(self, settings: Optional[Dict]) -> str:
        if not isinstance(settings, dict):
            return "cost"
        return self._objective_metric_from_metric_preference(
            settings.get("metric_preference", self.metric_preference)
        )

    @staticmethod
    def _benchmark_dataset_from_settings(settings: Optional[Dict]) -> str:
        if not isinstance(settings, dict):
            return "mt_bench"
        bench = str(settings.get("benchmark_dataset", "mt_bench")).strip().lower()
        if bench not in {"mt_bench", "gsm8k", "humaneval", "cnn_dailymail"}:
            return "mt_bench"
        return bench

    def _parse_host_port_from_endpoint(self, endpoint: str) -> tuple[str, int]:
        raw = str(endpoint or "").strip()
        if not raw:
            return (
                os.environ.get("AUTODRAFT_TARGET_HOST")
                or os.environ.get("AUTODRAFT_HOST")
                or "192.168.0.12",
                int(os.environ.get("AUTODRAFT_PORT", "26001")),
            )
        if "://" in raw:
            parsed = urlparse(raw)
            host = parsed.hostname or os.environ.get("AUTODRAFT_HOST", "127.0.0.1")
            port = parsed.port or int(os.environ.get("AUTODRAFT_PORT", "26001"))
            return host, int(port)
        stripped = raw.split("/", 1)[0]
        if ":" in stripped:
            host, port_s = stripped.rsplit(":", 1)
            try:
                return host.strip(), int(port_s.strip())
            except Exception:
                pass
        return stripped, int(os.environ.get("AUTODRAFT_PORT", "26001"))

    @staticmethod
    def _to_env_token(value: str) -> str:
        token = re.sub(r"[^A-Za-z0-9_]+", "_", str(value or "").strip().upper())
        token = re.sub(r"_+", "_", token).strip("_")
        # Keep env key names compact: drop trailing "_ADD" from UI labels.
        if token.endswith("_ADD"):
            token = token[:-4].strip("_")
        return token or "UNKNOWN"

    def _upsert_env_line(self, lines: List[str], key: str, value: str) -> None:
        line = f"{key}={value}"
        prefix = f"{key}="
        for idx, existing in enumerate(lines):
            if existing.startswith(prefix):
                lines[idx] = line
                return
        lines.append(line)

    def _remove_env_keys_by_prefix(self, lines: List[str], prefixes: List[str]) -> List[str]:
        if not prefixes:
            return lines
        return [line for line in lines if not any(line.startswith(prefix) for prefix in prefixes)]

    def _persist_server_env(self, spec) -> None:
        """
        Persist user-added server address into .env.
        Only HOST/PORT keys are stored per user-added server.
        """
        try:
            dotenv_path = self.repo_root / ".env"
            if dotenv_path.exists():
                lines = dotenv_path.read_text(encoding="utf-8").splitlines()
            else:
                lines = []

            server_token = self._to_env_token(getattr(spec, "name", ""))
            legacy_token = self._to_env_token(getattr(spec, "server_id", ""))
            block_header = f"## UI-added server: {getattr(spec, 'name', 'Unnamed server')}"
            host_key = f"AUTODRAFT_{server_token}_HOST"
            port_key = f"AUTODRAFT_{server_token}_PORT"
            old_style_add_token = f"{server_token}_ADD"
            block_keys = {}
            endpoint = str(getattr(spec, "endpoint", "")).strip()
            protocol = str(getattr(spec, "protocol", "")).strip().lower()
            if endpoint and protocol == "autodraft_target":
                host, port = self._parse_host_port_from_endpoint(endpoint)
                block_keys[host_key] = str(host).strip()
                block_keys[port_key] = str(int(port))

            lines = self._remove_env_keys_by_prefix(
                lines,
                [
                    f"AUTODRAFT_UI_SERVER_{legacy_token}_",
                    f"AUTODRAFT_UI_SERVER_{server_token}_",
                    f"AUTODRAFT_{server_token}_HOST=",
                    f"AUTODRAFT_{server_token}_PORT=",
                    f"AUTODRAFT_{old_style_add_token}_HOST=",
                    f"AUTODRAFT_{old_style_add_token}_PORT=",
                ],
            )

            if block_header not in lines:
                if lines and lines[-1].strip():
                    lines.append("")
                lines.append(block_header)
            for key, val in block_keys.items():
                self._upsert_env_line(lines, key, val)
                os.environ[key] = str(val)

            dotenv_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        except Exception:
            # Server add should still succeed even when .env persistence fails.
            return

    def _resolve_runtime_target_config(self, settings: Dict) -> Dict:
        host = (
            os.environ.get("AUTODRAFT_TARGET_HOST")
            or os.environ.get("AUTODRAFT_HOST")
            or "192.168.0.12"
        )
        port = int(os.environ.get("AUTODRAFT_PORT", "26001"))
        base_model = os.environ.get("AUTODRAFT_BASE_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
        # Keep profile/reference naming backward-compatible by default.
        server_name = os.environ.get("AUTODRAFT_SERVER_NAME", "rtxproa6000")
        selected_server_id = settings.get("selected_server_id")
        selected_model_id = settings.get("selected_model_id")
        selected_protocol = None
        if selected_server_id:
            spec = self.registry.get(str(selected_server_id))
            if spec is not None:
                selected_protocol = spec.protocol
                if spec.protocol == "autodraft_target":
                    host, port = self._parse_host_port_from_endpoint(spec.endpoint)
                    metadata_server_name = (
                        spec.metadata.get("server_name")
                        if isinstance(spec.metadata, dict)
                        else None
                    )
                    if metadata_server_name:
                        server_name = str(metadata_server_name)
                    elif spec.name:
                        server_name = str(spec.name)
                    is_bridge_external = bool(
                        isinstance(spec.metadata, dict) and spec.metadata.get("bridge_external")
                    )
                    if is_bridge_external:
                        bridge_runtime_base = (
                            spec.metadata.get("runtime_base_model")
                            if isinstance(spec.metadata, dict)
                            else None
                        )
                        base_model = str(
                            bridge_runtime_base
                            or os.environ.get("AUTODRAFT_BRIDGE_RUNTIME_BASE_MODEL")
                            or base_model
                        )
                    else:
                        if selected_model_id:
                            base_model = str(selected_model_id)
                        elif spec.default_model_id:
                            base_model = str(spec.default_model_id)
        return {
            "host": host,
            "port": int(port),
            "base_model": base_model,
            "server_name": server_name,
            "selected_protocol": selected_protocol,
        }

    def _resolve_device_name(self) -> str:
        configured = str(os.environ.get("AUTODRAFT_DEVICE_NAME", "")).strip()
        if configured:
            return configured
        # Auto-detect local draft runtime GPU name when env is not set.
        try:
            import torch  # Local import to avoid hard dependency at module import time.
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                name = str(torch.cuda.get_device_name(0)).strip()
                if name:
                    return name
        except Exception:
            pass
        return "unknown-gpu"

    def _resolve_selected_server_and_model(self, settings: Dict):
        selected_server_id = settings.get("selected_server_id")
        if not selected_server_id:
            return None, None, None
        spec = self.registry.get(str(selected_server_id))
        if spec is None:
            return None, None, f"Selected server not found: {selected_server_id}"
        selected_model_id = str(settings.get("selected_model_id") or "").strip()
        model = None
        if selected_model_id:
            wanted = selected_model_id.lower()
            for m in spec.models:
                if str(m.model_id or "").lower() == wanted:
                    model = m
                    break
        if model is None and spec.default_model_id:
            wanted = str(spec.default_model_id).lower()
            for m in spec.models:
                if str(m.model_id or "").lower() == wanted:
                    model = m
                    break
        if model is None and spec.models:
            model = spec.models[0]
        if model is None:
            return spec, None, f"No model available for server: {spec.name}"
        return spec, model, None

    def _is_server_only_api_selected(self, settings: Dict) -> bool:
        spec, _, _ = self._resolve_selected_server_and_model(settings)
        return bool(spec is not None and str(spec.protocol) == "openai_chat_completions")

    def _resolve_target_endpoint_for_control(self, settings: Dict) -> tuple[str, int, Optional[str]]:
        selected_server_id = settings.get("selected_server_id")
        if selected_server_id:
            spec = self.registry.get(str(selected_server_id))
            if spec is None:
                return "", 0, f"Selected server not found: {selected_server_id}"
            if spec.protocol != "autodraft_target":
                return "", 0, "Selected server is not autodraft_target."
            host, port = self._parse_host_port_from_endpoint(spec.endpoint)
            return str(host), int(port), None
        resolved = self._resolve_runtime_target_config(settings)
        return str(resolved["host"]), int(resolved["port"]), None

    def _target_control_sync(self, settings: Dict, payload: Dict, timeout_s: float = 20.0) -> Dict:
        host, port, err = self._resolve_target_endpoint_for_control(settings)
        if err:
            self._emit_runtime_log("target-control", f"resolve failed: {err}")
            return {"ok": False, "error": err}
        self._emit_runtime_log("target-control", f"connect {host}:{int(port)} payload={payload}")
        try:
            with socket.create_connection((host, int(port)), timeout=float(timeout_s)) as sock:
                send_json_with_size(sock, payload)
                reply, _ = recv_json_with_size(sock)
            self._emit_runtime_log("target-control", f"reply={reply}")
            if not isinstance(reply, dict):
                return {"ok": False, "error": "invalid_reply"}
            if str(reply.get("type")) == "error":
                return {"ok": False, "error": str(reply.get("message", "target_error")), "reply": reply}
            return {"ok": True, "reply": reply}
        except Exception as exc:
            self._emit_runtime_log("target-control", f"exception={exc}")
            return {"ok": False, "error": f"target_control_failed:{exc}"}

    async def ensure_remote_target_loaded(self, settings: Dict) -> Dict:
        if self._is_server_only_api_selected(settings):
            return {"ok": True, "message": "API mode does not use remote autodraft_target lifecycle."}
        resolved = self._resolve_runtime_target_config(settings)
        target_quantization = _normalize_quantization_mode(settings.get("target_quantization", "none"), default="none")
        quant_chain = _build_target_quantization_fallback_chain(target_quantization)
        attempts = max(1, int(os.environ.get("AUTODRAFT_TARGET_CONNECT_RETRIES", "8")))
        base_delay = max(0.05, float(os.environ.get("AUTODRAFT_TARGET_CONNECT_BACKOFF_SEC", "0.5")))
        unload_cooldown = max(0.0, float(os.environ.get("AUTODRAFT_TARGET_RELOAD_UNLOAD_COOLDOWN_SEC", "0.25")))
        last_error = "unknown_error"
        result = None
        connected = False
        for attempt in range(1, attempts + 1):
            self.phase = "connecting_target"
            self.phase_label = f"Connecting target ({attempt}/{attempts})..."
            status_result = await asyncio.to_thread(self._target_control_sync, settings, {"type": "status"}, 60.0)
            if status_result.get("ok"):
                connected = True
                break
            last_error = str(status_result.get("error", "unknown_error"))
            if self._is_connection_refused_error(last_error) and attempt < attempts:
                wait_s = min(5.0, base_delay * (2 ** (attempt - 1)))
                self.last_error = last_error
                self.last_line = f"Target connect retry in {wait_s:.1f}s ({attempt}/{attempts})..."
                await asyncio.sleep(wait_s)
                continue
            break
        if not connected:
            self.phase = "error"
            self.phase_label = "Target connect/load failed"
            return {
                "ok": False,
                "message": f"Failed to load target model: {last_error}",
            }
        attempted_quants: List[str] = []
        failures: List[str] = []
        for quant_mode in quant_chain:
            attempted_quants.append(str(quant_mode))
            self.phase = "target_loading"
            self.phase_label = f"Loading target model ({quant_mode})..."
            unload_result = await asyncio.to_thread(self._target_control_sync, settings, {"type": "unload_model"}, 30.0)
            if not unload_result.get("ok"):
                self._emit_runtime_log("target-control", f"fallback unload failed: {unload_result}")
            if unload_cooldown > 0:
                await asyncio.sleep(unload_cooldown)
            payload = {
                "type": "reload_model",
                "base_model_path": str(resolved["base_model"]),
                "quantization": str(quant_mode),
            }
            result = await asyncio.to_thread(self._target_control_sync, settings, payload, 180.0)
            if result.get("ok"):
                reply = result.get("reply") if isinstance(result.get("reply"), dict) else {}
                loaded = reply.get("loaded_model") or resolved["base_model"]
                loaded_quant = _normalize_quantization_mode(
                    reply.get("quantization", quant_mode), default=str(quant_mode)
                )
                changed = bool(reply.get("changed", True))
                mode = "reloaded" if changed else "already_loaded"
                fallback_applied = bool(loaded_quant != target_quantization or len(attempted_quants) > 1)
                self.phase = "target_loading"
                self.phase_label = "Target model loaded"
                return {
                    "ok": True,
                    "message": (
                        f"Target model {mode}: {loaded} ({loaded_quant}) "
                        f"[fallback attempted={attempted_quants}]"
                    ),
                    "reply": reply,
                    "fallback": {
                        "attempted_quantizations": attempted_quants,
                        "selected_quantization": loaded_quant,
                        "fallback_applied": fallback_applied,
                    },
                }
            last_error = str(result.get("error", "unknown_error"))
            failures.append(f"{quant_mode}:{last_error}")
            if not _is_target_memory_related_reload_error(last_error):
                break
        self.phase = "error"
        self.phase_label = "Target connect/load failed"
        guidance = (
            "Try a smaller target model, switch to a higher quantization level, "
            "or run on a larger-memory GPU."
        )
        return {
            "ok": False,
            "message": (
                f"Failed to load target model after quantization fallback. "
                f"attempted={attempted_quants}, failures={failures}. {guidance}"
            ),
        }

    async def unload_remote_target_model(self, settings: Dict) -> Dict:
        if self._is_server_only_api_selected(settings):
            return {"ok": True, "message": "API mode: no target model to unload."}
        attempts = max(1, int(os.environ.get("AUTODRAFT_TARGET_CONNECT_RETRIES", "8")))
        base_delay = max(0.05, float(os.environ.get("AUTODRAFT_TARGET_CONNECT_BACKOFF_SEC", "0.5")))
        last_error = "unknown_error"
        for attempt in range(1, attempts + 1):
            result = await asyncio.to_thread(
                self._target_control_sync, settings, {"type": "unload_model"}, 30.0
            )
            if result.get("ok"):
                return {"ok": True, "message": "Target model unloaded."}
            last_error = str(result.get("error", "unknown_error"))
            if self._is_connection_refused_error(last_error):
                if attempt < attempts:
                    await asyncio.sleep(min(5.0, base_delay * (2 ** (attempt - 1))))
                    continue
                # Target is not accepting control connection; treat as effectively unloaded.
                return {
                    "ok": True,
                    "message": "Target control endpoint unreachable; treated as unloaded.",
                }
            break
        return {"ok": False, "message": f"Target unload failed: {last_error}"}

    async def shutdown_remote_target(self, settings: Dict) -> Dict:
        if self._is_server_only_api_selected(settings):
            return {"ok": False, "message": "Server-only API mode has no remote autodraft_target process to shutdown."}
        result = await asyncio.to_thread(
            self._target_control_sync, settings, {"type": "shutdown"}, 20.0
        )
        if not result.get("ok"):
            return {"ok": False, "message": f"Target shutdown failed: {result.get('error', 'unknown_error')}"}
        rtype = str((result.get("reply") or {}).get("type", ""))
        if rtype not in {"bye", "shutdown_ok"}:
            return {"ok": False, "message": f"Unexpected target shutdown reply: {rtype or 'unknown'}"}
        return {"ok": True, "message": "Target server shutdown requested."}

    def _validate_execution_server(self, settings: Dict) -> Optional[str]:
        selected_server_id = settings.get("selected_server_id")
        if not selected_server_id:
            return None
        spec = self.registry.get(str(selected_server_id))
        if spec is None:
            return f"Selected server not found: {selected_server_id}"
        algorithm = str(settings.get("algorithm", "AutoDraft"))
        if spec.protocol == "openai_chat_completions":
            if algorithm != "Server-Only-AR":
                return (
                    "Selected server is an external Chat API and supports Server-only AR mode only. "
                    "Hybrid speculative decoding requires autodraft_target internal signals "
                    "(verify path / logits / KV state). Please switch algorithm to Server-Only-AR."
                )
            if spec.requires_api_key and not spec.api_key:
                return f"API key required for server: {spec.name}"
            if not str(spec.endpoint or "").strip():
                return f"Missing endpoint for server: {spec.name}"
            _, model, model_err = self._resolve_selected_server_and_model(settings)
            if model_err:
                return model_err
            if model is None:
                return f"No model available for server: {spec.name}"
            return None
        if spec.protocol != "autodraft_target":
            return (
                f"Unsupported server protocol for execution: {spec.protocol}. "
                "Use autodraft_target (hybrid-capable target) or openai_chat_completions (Server-only API)."
            )
        if algorithm in {"Server-Only", "Server-Only-AR"}:
            return None
        target_model = _canonical_model_key(settings.get("selected_model_id"))
        draft_model = _canonical_model_key(settings.get("draft_model_path"))
        if target_model and draft_model:
            allow = TARGET_DRAFT_COMPATIBILITY.get(target_model)
            if not allow:
                return (
                    f"No compatibility rule for target '{target_model}'. "
                    "Please choose one of the supported target models."
                )
            if draft_model not in allow:
                return (
                    f"Incompatible draft model for target '{target_model}'. "
                    f"Allowed draft models: {', '.join(sorted(allow))}"
                )
        return None

    def _make_openai_chat_payload(self, model_id: str, text: str, max_new_tokens: int) -> Dict:
        return {
            "model": str(model_id),
            "messages": [{"role": "user", "content": str(text)}],
            "temperature": 0.0,
            "max_tokens": int(max_new_tokens),
        }

    def _extract_openai_reply_text(self, out: Dict) -> str:
        if not isinstance(out, dict):
            return ""
        choices = out.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        txt = item.get("text")
                        if isinstance(txt, str):
                            parts.append(txt)
                return "".join(parts)
        text = first.get("text")
        if isinstance(text, str):
            return text
        return ""

    def _api_server_only_chat_sync(self, user_text: str, settings: Dict) -> Dict:
        spec, model, err = self._resolve_selected_server_and_model(settings)
        if err:
            return {"ok": False, "error": err}
        if spec is None or model is None:
            return {"ok": False, "error": "No API server/model selected."}
        timeout_s = 60.0
        max_new_tokens = int(settings.get("max_new_tokens", 512))
        max_new_tokens = max(1, min(4096, max_new_tokens))
        payload = self._make_openai_chat_payload(model.model_id, user_text, max_new_tokens)
        t0 = time.perf_counter()
        out = self.probe_runner._post_json(spec.endpoint, payload, spec.api_key, timeout_s)
        elapsed = max(1e-6, time.perf_counter() - t0)
        reply = self._extract_openai_reply_text(out)
        usage = out.get("usage", {}) if isinstance(out, dict) else {}
        usage = usage if isinstance(usage, dict) else {}
        completion_tokens = float(usage.get("completion_tokens", 0) or 0)
        prompt_tokens = float(usage.get("prompt_tokens", 0) or 0)
        if completion_tokens <= 0:
            completion_tokens = max(1.0, float(len(reply.split())))
        throughput = completion_tokens / elapsed if elapsed > 0 else 0.0
        cost_per_1m = self.probe_runner._estimate_cost_per_1m(usage, model)
        target_energy = self.probe_runner._estimate_energy_per_1m(max(throughput, 1e-9), spec)
        total_tokens = max(1.0, completion_tokens + prompt_tokens)
        if cost_per_1m is None:
            cost_per_1m = 0.0
        stats = {
            "gpu_energy": float(target_energy or 0.0),
            "draft_cost": 0.0,
            "target_cost": float(cost_per_1m),
            "throughput": float(throughput),
            "cost_per_1m": float(cost_per_1m),
            "total_cost_per_1m": float(cost_per_1m),
            "api_cost_per_1m": float(cost_per_1m),
            "draft_energy_per_1m": 0.0,
            "target_energy_per_1m_kwh": float(target_energy) if target_energy is not None else None,
            "api_total_tokens": float(total_tokens),
        }
        return {"ok": True, "reply": reply, "stats": stats}

    def _api_server_only_benchmark_sync(self, settings: Dict) -> Dict:
        spec, model, err = self._resolve_selected_server_and_model(settings)
        if err:
            return {"ok": False, "error": err}
        if spec is None or model is None:
            return {"ok": False, "error": "No API server/model selected."}
        req = ProbeRequest(
            prompt="Summarize speculative decoding in one sentence.",
            max_tokens=max(1, min(256, int(settings.get("max_new_tokens", 64)))),
            temperature=0.0,
            timeout_s=30.0,
            warmup_runs=0,
            measured_runs=1,
        )
        result = self.probe_runner.probe_server_model(spec, model, req)
        if not result.get("ok"):
            return {"ok": False, "error": str(result.get("error", "probe_failed"))}
        cost_per_1m = float(result.get("metric_per_1m") or 0.0)
        throughput = float(result.get("throughput_tps") or 0.0)
        target_energy = result.get("target_energy_per_1m_kwh")
        stats = {
            "gpu_energy": float(target_energy or 0.0),
            "draft_cost": 0.0,
            "target_cost": cost_per_1m,
            "throughput": throughput,
            "cost_per_1m": cost_per_1m,
            "total_cost_per_1m": cost_per_1m,
            "api_cost_per_1m": cost_per_1m,
            "draft_energy_per_1m": 0.0,
        }
        return {"ok": True, "stats": stats}

    def _update_phase_from_line(self, line: str):
        l = (line or "").lower()
        if "[startup] target_connecting" in l:
            self.phase = "connecting_target"
            self.phase_label = "Connecting target..."
            return
        if "[startup] target_connected" in l:
            self.phase = "target_connected"
            self.phase_label = "Target connected"
            return
        if "[startup] target_reloading" in l:
            self.phase = "target_loading"
            self.phase_label = "Loading target model..."
            return
        if "[startup] target_reload_ok" in l or "[startup] target_already_loaded" in l:
            self.phase = "target_loaded"
            self.phase_label = "Target model ready"
            return
        if ("Target profiling file is missing. Requesting automatic generation" in l) or (
            "target profile" in l and "auto" in l and ("start" in l or "starting" in l)
        ):
            self.phase = "target_profiling_wait"
            self.phase_label = "Waiting for target profiling..."
            return
        if "loading checkpoint shards" in l:
            if self._is_reference_phase():
                self.phase = "reference_loading"
                self.phase_label = "Refreshing reference (loading model)..."
            else:
                self.phase = "loading_model"
                self.phase_label = "Loading model..."
            return
        if "[warmup] running warmup" in l:
            if self._is_reference_phase():
                self.phase = "reference_warmup"
                self.phase_label = "Refreshing reference (warmup)..."
            else:
                self.phase = "warmup"
                self.phase_label = "Warmup..."
            return
        if self._is_reference_phase() and ("profiling width=" in line or "profiling width=" in l):
            self.phase = "reference_profiling"
            self.phase_label = "Refreshing reference (profiling draft runtime)..."
            return
        if "[reference test]" in l:
            self.phase = "reference_running"
            self.phase_label = "Refreshing reference..."
            return
        if "[reference]" in l and "no cache found" in l:
            self.phase = "reference"
            self.phase_label = "Reference measuring..."
            return
        if "[reference]" in l and "loaded cached reference" in l:
            self.phase = "reference_cached"
            self.phase_label = "Reference loaded"
            return
        if "[reference]" in l and "saved reference cache" in l:
            self.phase = "reference_done"
            self.phase_label = "Reference done"
            return
        if l.startswith("warmup done"):
            # Keep current phase unless nothing meaningful is set.
            if self.phase in {"idle", "loading_model"}:
                self.phase = "warmup_done"
                self.phase_label = "Warmup done"
            return
        if "run completed successfully" in l:
            self.phase = "completed"
            self.phase_label = "Completed"
            return
        if "run failed" in l:
            self.phase = "error"
            self.phase_label = "Failed"
            return

    def _build_base_command(self, settings: Dict) -> list[str]:
        algorithm = settings.get("algorithm", "AutoDraft")
        proactive = bool(settings.get("proactive_drafting", False))
        if algorithm in {"Server-Only", "Server-Only-AR"}:
            proactive = False
        objective_mode = str(settings.get("objective_selection_mode", "blend")).lower()
        objective_mode = "constraint" if objective_mode == "constraint" else "blend"
        constraint_target = str(settings.get("constraint_target", "metric")).lower()
        constraint_target = "tps" if constraint_target == "tps" else "metric"
        objective_metric = self._objective_metric_from_settings(settings)
        metric_constraint_per_1m_token = float(settings.get("metric_constraint_per_1m_token", 14.0))
        min_tps_constraint = float(settings.get("min_tps_constraint", 0.0) or 0.0)
        online_profile_update = bool(settings.get("online_profile_update", True))
        try:
            online_profile_lr = float(settings.get("online_profile_lr", 0.05))
        except Exception:
            online_profile_lr = 0.05
        cs = float(settings.get("cost", 0.15))
        cs = max(0.0, min(1.0, cs))
        target_quantization = _normalize_quantization_mode(settings.get("target_quantization", "none"), default="none")
        draft_quantization = _normalize_quantization_mode(settings.get("draft_quantization", "none"), default="none")

        resolved = self._resolve_runtime_target_config(settings)
        # Keep defaults overridable by env for easy deployment.
        host = resolved["host"]
        port = resolved["port"]
        base_model = resolved["base_model"]
        draft_model = settings.get("draft_model_path") or os.environ.get(
            "AUTODRAFT_DRAFT_MODEL", "meta-llama/Llama-3.2-3B-Instruct"
        )
        tokenizer_model = os.environ.get("AUTODRAFT_TOKENIZER_PATH", "")
        device_map = os.environ.get("AUTODRAFT_DEVICE_MAP", "cuda:0")
        device_name = self._resolve_device_name()
        server_name = resolved["server_name"]
        bench_name = self._benchmark_dataset_from_settings(settings)

        cmd = [
            "python3",
            "-u",
            "-m",
            "evaluation.eval_autodraft_draft",
            "--host",
            host,
            "--port",
            str(port),
            "--base-model-path",
            base_model,
            "--draft-model-path",
            draft_model,
            "--bench-name",
            bench_name,
            "--temperature",
            "0.0",
            "--nodes",
            "150",
            "--max_depth",
            "15",
            "--device-map",
            device_map,
            "--num-choices",
            "1",
            "--device-name",
            device_name,
            "--server-name",
            str(server_name),
            "--draft-per-hour-cost",
            "0.152",
            "--target-per-hour-cost",
            "1.208",
            "--user-communication-cost-per-gb",
            "2.3333333333",
            "--accept-length-margin",
            "0.05",
            "--enable-gpu-monitor",
            "--target-quantization",
            target_quantization,
            "--seed",
            "4",
            "--objective-metric",
            objective_metric,
            "--objective-selection-mode",
            objective_mode,
            "--constraint-target",
            constraint_target,
            "--cost-sensitivity",
            str(cs),
            "--online-profile-lr",
            str(online_profile_lr),
        ]
        if not online_profile_update:
            cmd.append("--disable-online-profile-update")
        if draft_quantization == "4bit":
            cmd.append("--load-in-4bit")
        elif draft_quantization == "8bit":
            cmd.append("--load-in-8bit")
        if algorithm == "Server-Only":
            cmd.append("--force-server-only")
        elif algorithm == "Server-Only-AR":
            # Approximate strict autoregressive mode by forcing server-only with single-node/single-depth tree.
            cmd.extend(
                [
                    "--force-server-only",
                    "--nodes",
                    "1",
                    "--max_depth",
                    "1",
                    "--fixed-width",
                    "--fixed-width-value",
                    "1",
                    "--fixed-nnodes",
                    "--fixed-depth",
                ]
            )
        else:
            cmd.append("--disable-server-only")

        if proactive:
            cmd.extend(
                [
                    "--proactive-drafting",
                    "--adaptive-proactive-threshold",
                    "--proactive-threshold",
                    "0.0",
                ]
            )

        if algorithm == "OPT-Tree":
            cmd.append("--opt-tree")
        elif algorithm == "Fixed-tree":
            cmd.extend(["--fixed-depth", "--fixed-nnodes", "--fixed-width"])
        if tokenizer_model:
            cmd.extend(["--tokenizer-path", tokenizer_model])
        if objective_mode == "constraint":
            if constraint_target == "metric":
                cmd.extend(["--metric-constraint-per-1m-token", str(metric_constraint_per_1m_token)])
            elif min_tps_constraint > 0:
                cmd.extend(["--min-tps-constraint", str(min_tps_constraint)])
        return cmd

    def _build_benchmark_command(self, settings: Dict) -> list[str]:
        cmd = self._build_base_command(settings)
        cmd.extend(["--limit", "1"])
        return cmd

    def _build_chat_command(self, settings: Dict, question_file: Path, answer_file: Path) -> list[str]:
        cmd = self._build_base_command(settings)
        cmd.extend(
            [
                "--question-file",
                str(question_file),
                "--answer-file",
                str(answer_file),
                "--limit",
                "1",
                "--num-choices",
                "1",
            ]
        )
        return cmd

    def _build_persistent_chat_command(self, settings: Dict) -> list[str]:
        cmd = self._build_base_command(settings)
        cmd.extend(
            [
                "--chat-mode",
                "--chat-max-new-tokens",
                "512",
            ]
        )
        return cmd

    @staticmethod
    def _norm_reference_token(s: str) -> str:
        out = []
        for ch in str(s).split("/")[-1].lower():
            out.append(ch if ("a" <= ch <= "z") or ("0" <= ch <= "9") else "-")
        v = "".join(out).strip("-")
        while "--" in v:
            v = v.replace("--", "-")
        return v or "none"

    def _read_latest_reference_payload(self, settings: Dict) -> Optional[Dict]:
        try:
            resolved = self._resolve_runtime_target_config(settings)
            base_model = resolved["base_model"]
            draft_model = settings.get("draft_model_path") or os.environ.get(
                "AUTODRAFT_DRAFT_MODEL", "meta-llama/Llama-3.2-3B-Instruct"
            )
            device_name = self._resolve_device_name()
            server_name = resolved["server_name"]
            metric = self._objective_metric_from_settings(settings)
            bench = self._benchmark_dataset_from_settings(settings)
            objective_mode = str(settings.get("objective_selection_mode", "blend")).lower()
            objective_mode = "constraint" if objective_mode == "constraint" else "blend"
            target_quantization = _normalize_quantization_mode(settings.get("target_quantization", "none"), default="none")
            draft_quantization = _normalize_quantization_mode(settings.get("draft_quantization", "none"), default="none")
            pattern = (
                f"ref_{self._norm_reference_token(server_name)}_{self._norm_reference_token(base_model)}_"
                f"{self._norm_reference_token(device_name)}_{self._norm_reference_token(draft_model)}_"
                f"tq-{self._norm_reference_token(target_quantization)}_dq-{self._norm_reference_token(draft_quantization)}_"
                f"{bench}_{metric}_{objective_mode}_*.json"
            )
            ref_dir = self.repo_root / "data" / "reference"
            candidates = sorted(ref_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
            if not candidates:
                # backward compatibility: old filename without quantization tokens
                legacy_pattern = (
                    f"ref_{self._norm_reference_token(server_name)}_{self._norm_reference_token(base_model)}_"
                    f"{self._norm_reference_token(device_name)}_{self._norm_reference_token(draft_model)}_"
                    f"{bench}_{metric}_{objective_mode}_*.json"
                )
                candidates = sorted(ref_dir.glob(legacy_pattern), key=lambda p: p.stat().st_mtime, reverse=True)
            if not candidates:
                # oldest fallback: mode token absent
                legacy_pattern2 = (
                    f"ref_{self._norm_reference_token(server_name)}_{self._norm_reference_token(base_model)}_"
                    f"{self._norm_reference_token(device_name)}_{self._norm_reference_token(draft_model)}_"
                    f"{bench}_{metric}_*.json"
                )
                candidates = sorted(ref_dir.glob(legacy_pattern2), key=lambda p: p.stat().st_mtime, reverse=True)
            if not candidates:
                return None
            return json.loads(candidates[0].read_text(encoding="utf-8"))
        except Exception:
            return None

    def _curve_points_from_reference_payload(self, payload: Optional[Dict], objective_mode: str) -> List[Dict]:
        if not isinstance(payload, dict):
            return []
        mode = "constraint" if str(objective_mode).lower() == "constraint" else "blend"
        constraint_target = str(payload.get("constraint_target", "metric")).lower()
        if constraint_target not in {"metric", "tps"}:
            constraint_target = "metric"
        rows = []
        if mode == "constraint":
            rows = payload.get("reference_tradeoff_curve_by_constraint")
            if not isinstance(rows, list) or not rows:
                rows = payload.get("reference_constraint_anchor_curve")
        else:
            rows = payload.get("reference_tradeoff_curve_cs0_1")
            if not isinstance(rows, list) or not rows:
                rows = payload.get("reference_cs_anchor_curve")
        if not isinstance(rows, list):
            return []
        points = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            try:
                if mode == "constraint":
                    if constraint_target == "tps":
                        selector = float(r.get("min_tps_constraint", r.get("predicted_tps")))
                    else:
                        selector = float(r.get("metric_constraint_per_1m_token"))
                else:
                    selector = float(r.get("cost_sensitivity"))
                metric_per_1m = float(
                    r.get(
                        "predicted_metric_per_1m_token",
                        r.get("predicted_objective_per_1m_token"),
                    )
                )
                tps = float(r.get("predicted_tps"))
            except Exception:
                continue
            if not (
                math.isfinite(selector)
                and math.isfinite(metric_per_1m)
                and math.isfinite(tps)
            ):
                continue
            points.append(
                {
                    "selector_value": selector,
                    "metric_per_1m": metric_per_1m,
                    "throughput_tps": tps,
                }
            )
        return points

    def _curve_points_from_runtime_state(self, objective_mode: str) -> List[Dict]:
        payload = {
            "constraint_target": self.current_settings.get("constraint_target", "metric"),
            "reference_tradeoff_curve_cs0_1": self.reference_tradeoff_curve_cs0_1,
            "reference_tradeoff_curve_by_constraint": self.reference_tradeoff_curve_by_constraint,
            "reference_cs_anchor_curve": self.reference_cs_anchor_curve,
            "reference_constraint_anchor_curve": self.reference_constraint_anchor_curve,
        }
        return self._curve_points_from_reference_payload(payload, objective_mode)

    def _load_tradeoff_from_latest_reference(self, settings: Dict):
        try:
            data = self._read_latest_reference_payload(settings)
            if not isinstance(data, dict):
                self.feasible_constraint_range_per_1m = None
                self.feasible_tps_range = None
                self.reference_tradeoff_curve_cs0_1 = None
                self.reference_tradeoff_curve_by_constraint = None
                return
            fm = data.get("feasible_metric_per_token")
            if isinstance(fm, dict) and fm.get("min") is not None and fm.get("max") is not None:
                self.feasible_constraint_range_per_1m = {
                    "min": float(fm["min"]) * 1_000_000.0,
                    "max": float(fm["max"]) * 1_000_000.0,
                }
            else:
                self.feasible_constraint_range_per_1m = None
            ft = data.get("feasible_tps")
            if isinstance(ft, dict) and ft.get("min") is not None and ft.get("max") is not None:
                self.feasible_tps_range = {
                    "min": float(ft["min"]),
                    "max": float(ft["max"]),
                }
            else:
                self.feasible_tps_range = None
            self.reference_tradeoff_curve_cs0_1 = (
                data.get("reference_tradeoff_curve_cs0_1")
                if isinstance(data.get("reference_tradeoff_curve_cs0_1"), list)
                else None
            )
            self.reference_tradeoff_curve_by_constraint = (
                data.get("reference_tradeoff_curve_by_constraint")
                if isinstance(data.get("reference_tradeoff_curve_by_constraint"), list)
                else None
            )
            self.reference_cs_anchor_curve = (
                data.get("reference_cs_anchor_curve")
                if isinstance(data.get("reference_cs_anchor_curve"), list)
                else None
            )
            self.reference_constraint_anchor_curve = (
                data.get("reference_constraint_anchor_curve")
                if isinstance(data.get("reference_constraint_anchor_curve"), list)
                else None
            )
        except Exception:
            self.feasible_constraint_range_per_1m = None
            self.feasible_tps_range = None
            self.reference_tradeoff_curve_cs0_1 = None
            self.reference_tradeoff_curve_by_constraint = None
            self.reference_cs_anchor_curve = None
            self.reference_constraint_anchor_curve = None

    def _parse_answer_file(self, answer_file: Path) -> Optional[str]:
        if not answer_file.exists():
            return None
        try:
            lines = answer_file.read_text(encoding="utf-8").strip().splitlines()
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
        except Exception:
            return None

    async def _stream_reader(self, stream: asyncio.StreamReader, is_stderr: bool = False):
        while True:
            line_b = await stream.readline()
            if not line_b:
                break
            line = line_b.decode("utf-8", errors="ignore").replace("\r", " ").rstrip("\n").strip()
            self._emit_runtime_log("draft-stderr" if is_stderr else "draft-stdout", line)
            self.last_line = line
            self._update_phase_from_line(line)
            if is_stderr and self._looks_like_error_line(line) and not self._looks_like_progress_line(line):
                self.last_error = line
            m = LINE_TPS_RE.search(line)
            if m:
                try:
                    self.stats["cost_per_1m"] = float(m.group(1))
                    self.stats["throughput"] = float(m.group(2))
                    # Cheap proxy for chart continuity.
                    self.stats["draft_cost"] = self.stats["cost_per_1m"] * 0.4
                    self.stats["target_cost"] = self.stats["cost_per_1m"] * 0.6
                except Exception:
                    pass
            m2 = LINE_ENERGY_RE.search(line)
            if m2:
                try:
                    self.stats["gpu_energy"] = float(m2.group(1))
                    self.stats["draft_energy_per_1m"] = float(m2.group(2))
                except Exception:
                    pass
            if "Original:" in line:
                p = line.split("Original:", 1)[-1].strip()
                if p:
                    self.last_result_json = p

    async def _chat_stream_reader(self, stream: asyncio.StreamReader, is_stderr: bool = False):
        while True:
            line_b = await stream.readline()
            if not line_b:
                break
            line = line_b.decode("utf-8", errors="ignore").replace("\r", " ").rstrip("\n").strip()
            self._emit_runtime_log("chat-stderr" if is_stderr else "chat-stdout", line)
            self.last_line = line
            self._update_phase_from_line(line)
            if is_stderr and self._looks_like_error_line(line) and not self._looks_like_progress_line(line):
                self.last_error = line
            if not line.startswith("CHAT_JSON:"):
                continue
            payload_str = line.split("CHAT_JSON:", 1)[-1].strip()
            try:
                payload = json.loads(payload_str)
            except Exception:
                continue
            p_type = str(payload.get("type", "")).lower()
            if p_type == "ready":
                self.chat_ready = True
                self.phase = "chat_ready"
                self.phase_label = "Chat ready"
                f_range = payload.get("feasible_constraint_range_per_1m", None)
                if isinstance(f_range, dict):
                    try:
                        self.feasible_constraint_range_per_1m = {
                            "min": float(f_range.get("min")),
                            "max": float(f_range.get("max")),
                        }
                    except Exception:
                        self.feasible_constraint_range_per_1m = None
                tps_range = payload.get("feasible_tps_range", None)
                if isinstance(tps_range, dict):
                    try:
                        self.feasible_tps_range = {
                            "min": float(tps_range.get("min")),
                            "max": float(tps_range.get("max")),
                        }
                    except Exception:
                        self.feasible_tps_range = None
                t_curve_cs = payload.get("reference_tradeoff_curve_cs0_1", None)
                self.reference_tradeoff_curve_cs0_1 = t_curve_cs if isinstance(t_curve_cs, list) else None
                t_curve_cons = payload.get("reference_tradeoff_curve_by_constraint", None)
                self.reference_tradeoff_curve_by_constraint = t_curve_cons if isinstance(t_curve_cons, list) else None
                cs_anchor = payload.get("reference_cs_anchor_curve", None)
                self.reference_cs_anchor_curve = cs_anchor if isinstance(cs_anchor, list) else None
                cons_anchor = payload.get("reference_constraint_anchor_curve", None)
                self.reference_constraint_anchor_curve = cons_anchor if isinstance(cons_anchor, list) else None
            elif p_type == "error":
                self.phase = "chat_error"
                self.phase_label = "Chat error"
                self.last_error = str(payload.get("message", self.last_error))
            elif p_type == "chat_partial":
                pstats = payload.get("stats", {}) if isinstance(payload, dict) else {}
                if isinstance(pstats, dict):
                    try:
                        if "gpu_energy" in pstats:
                            self.stats["gpu_energy"] = float(pstats.get("gpu_energy", self.stats["gpu_energy"]))
                        if "draft_cost" in pstats:
                            self.stats["draft_cost"] = float(pstats.get("draft_cost", self.stats["draft_cost"]))
                        if "target_cost" in pstats:
                            self.stats["target_cost"] = float(pstats.get("target_cost", self.stats["target_cost"]))
                        if "throughput" in pstats:
                            self.stats["throughput"] = float(pstats.get("throughput", self.stats["throughput"]))
                    except Exception:
                        pass
            elif p_type == "chat_reply":
                final_stats = payload.get("final_stats", {}) if isinstance(payload, dict) else {}
                if isinstance(final_stats, dict):
                    try:
                        if "gpu_energy" in final_stats:
                            self.stats["gpu_energy"] = float(final_stats.get("gpu_energy", self.stats["gpu_energy"]))
                        if "draft_cost" in final_stats:
                            self.stats["draft_cost"] = float(final_stats.get("draft_cost", self.stats["draft_cost"]))
                        if "target_cost" in final_stats:
                            self.stats["target_cost"] = float(final_stats.get("target_cost", self.stats["target_cost"]))
                        if "throughput" in final_stats:
                            self.stats["throughput"] = float(final_stats.get("throughput", self.stats["throughput"]))
                    except Exception:
                        pass
                f_range = payload.get("feasible_constraint_range_per_1m", None)
                if isinstance(f_range, dict):
                    try:
                        self.feasible_constraint_range_per_1m = {
                            "min": float(f_range.get("min")),
                            "max": float(f_range.get("max")),
                        }
                    except Exception:
                        self.feasible_constraint_range_per_1m = None
                tps_range = payload.get("feasible_tps_range", None)
                if isinstance(tps_range, dict):
                    try:
                        self.feasible_tps_range = {
                            "min": float(tps_range.get("min")),
                            "max": float(tps_range.get("max")),
                        }
                    except Exception:
                        self.feasible_tps_range = None
                t_curve_cs = payload.get("reference_tradeoff_curve_cs0_1", None)
                if isinstance(t_curve_cs, list):
                    self.reference_tradeoff_curve_cs0_1 = t_curve_cs
                t_curve_cons = payload.get("reference_tradeoff_curve_by_constraint", None)
                if isinstance(t_curve_cons, list):
                    self.reference_tradeoff_curve_by_constraint = t_curve_cons
                cs_anchor = payload.get("reference_cs_anchor_curve", None)
                if isinstance(cs_anchor, list):
                    self.reference_cs_anchor_curve = cs_anchor
                cons_anchor = payload.get("reference_constraint_anchor_curve", None)
                if isinstance(cons_anchor, list):
                    self.reference_constraint_anchor_curve = cons_anchor
            await self._chat_queue.put(payload)

    async def _run(self, settings: Dict):
        cmd = self._build_benchmark_command(settings)
        self.current_settings = dict(settings)
        self.exit_code = None
        self.last_error = ""
        self.last_line = "Starting AutoDraft run..."
        self.phase = "starting"
        self.phase_label = "Starting..."
        self.running = True
        self.stats["running"] = True
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(self.repo_root),
            env={**os.environ, "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python"},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert self._proc.stdout is not None
        assert self._proc.stderr is not None
        self._stdout_task = asyncio.create_task(self._stream_reader(self._proc.stdout, is_stderr=False))
        self._stderr_task = asyncio.create_task(self._stream_reader(self._proc.stderr, is_stderr=True))
        await self._proc.wait()
        await asyncio.gather(self._stdout_task, self._stderr_task, return_exceptions=True)
        self.exit_code = self._proc.returncode
        self.running = False
        self.stats["running"] = False
        if self.exit_code == 0:
            self.last_line = "Run completed successfully."
            self.phase = "completed"
            self.phase_label = "Completed"
        else:
            self.last_line = f"Run failed (exit_code={self.exit_code})."
            self.phase = "error"
            self.phase_label = "Failed"

    async def _run_api_server_only(self, settings: Dict):
        self.current_settings = dict(settings)
        self.exit_code = None
        self.last_error = ""
        self.last_line = "Starting Server-only API benchmark..."
        self.phase = "starting"
        self.phase_label = "Starting..."
        self.running = True
        self.stats["running"] = True
        try:
            result = await asyncio.to_thread(self._api_server_only_benchmark_sync, settings)
            if not result.get("ok"):
                self.exit_code = 1
                self.last_line = f"Server-only API benchmark failed: {result.get('error', 'unknown_error')}"
                self.last_error = str(result.get("error", "unknown_error"))
                self.phase = "error"
                self.phase_label = "Failed"
                return
            bench_stats = result.get("stats") if isinstance(result.get("stats"), dict) else {}
            if bench_stats:
                self.stats["gpu_energy"] = float(bench_stats.get("gpu_energy", self.stats["gpu_energy"]))
                self.stats["draft_cost"] = float(bench_stats.get("draft_cost", self.stats["draft_cost"]))
                self.stats["target_cost"] = float(bench_stats.get("target_cost", self.stats["target_cost"]))
                self.stats["throughput"] = float(bench_stats.get("throughput", self.stats["throughput"]))
                self.stats["cost_per_1m"] = float(bench_stats.get("cost_per_1m", self.stats["cost_per_1m"]))
                self.stats["draft_energy_per_1m"] = float(
                    bench_stats.get("draft_energy_per_1m", self.stats["draft_energy_per_1m"])
                )
            self.exit_code = 0
            self.last_line = "Server-only API benchmark completed successfully."
            self.phase = "completed"
            self.phase_label = "Completed"
        except asyncio.CancelledError:
            self.exit_code = 130
            self.last_line = "Server-only API benchmark cancelled."
            self.phase = "stopped"
            self.phase_label = "Stopped"
            raise
        finally:
            self.running = False
            self.stats["running"] = False

    async def start_run(self, settings: Dict) -> str:
        async with self._lock:
            err = self._validate_execution_server(settings)
            if err:
                return err
            if self.running:
                return "A run is already in progress. Send 'stop' first."
            if self._is_server_only_api_selected(settings):
                self._run_task = asyncio.create_task(self._run_api_server_only(settings))
                return "Started Server-only API benchmark run."
            preload = await self.ensure_remote_target_loaded(settings)
            if not preload.get("ok"):
                return str(preload.get("message", "Failed to prepare target runtime."))
            self._run_task = asyncio.create_task(self._run(settings))
            return f"Started AutoDraft run. {preload.get('message', '')}".strip()

    async def stop_run(self, settings: Optional[Dict] = None) -> str:
        async with self._lock:
            if not self.running:
                if settings and not self._is_server_only_api_selected(settings):
                    unload = await self.unload_remote_target_model(settings)
                    if unload.get("ok"):
                        self._reset_tradeoff_state()
                        return "No active run. Target model unloaded."
                self._reset_tradeoff_state()
                return "No active run."
            msg = "Stopping current run..."
            if self._proc is not None:
                self._proc.terminate()
            elif self._run_task is not None and not self._run_task.done():
                self._run_task.cancel()
            if settings and not self._is_server_only_api_selected(settings):
                unload = await self.unload_remote_target_model(settings)
                if unload.get("ok"):
                    msg = f"{msg} Target model unloaded."
                else:
                    msg = f"{msg} {unload.get('message', '')}".strip()
            self._reset_tradeoff_state()
            return msg

    async def refresh_reference_cache(self, settings: Dict, detailed_profile: bool = False) -> str:
        async with self._lock:
            err = self._validate_execution_server(settings)
            if err:
                return err
            if self._is_server_only_api_selected(settings):
                return (
                    "Reference cache is only used for hybrid-capable autodraft_target runtime. "
                    "Server-only API mode does not use local reference cache."
                )
            if self.running:
                return "Benchmark is running. Stop benchmark first."
            # If chat runtime is already alive with the same runtime key, avoid launching
            # another draft process that competes for the same target socket.
            current_key = self._settings_key(settings)
            chat_alive_same_runtime = (
                self._chat_proc is not None
                and self._chat_proc.returncode is None
                and self.chat_ready
                and self._chat_settings_key == current_key
            )
            if chat_alive_same_runtime and not detailed_profile:
                self.phase = "reference_refresh"
                self.phase_label = "Refreshing reference..."
                self.last_line = "Reusing active chat runtime reference data..."
                self.last_error = ""
                # Prefer in-memory curve already reported by chat runtime; if missing, try disk cache.
                has_curve_in_memory = bool(
                    (isinstance(self.reference_tradeoff_curve_cs0_1, list) and len(self.reference_tradeoff_curve_cs0_1) > 0)
                    or (isinstance(self.reference_tradeoff_curve_by_constraint, list) and len(self.reference_tradeoff_curve_by_constraint) > 0)
                    or (isinstance(self.reference_cs_anchor_curve, list) and len(self.reference_cs_anchor_curve) > 0)
                    or (
                        isinstance(self.reference_constraint_anchor_curve, list)
                        and len(self.reference_constraint_anchor_curve) > 0
                    )
                )
                if not has_curve_in_memory:
                    self._load_tradeoff_from_latest_reference(settings)
                    has_curve_in_memory = bool(
                        (isinstance(self.reference_tradeoff_curve_cs0_1, list) and len(self.reference_tradeoff_curve_cs0_1) > 0)
                        or (
                            isinstance(self.reference_tradeoff_curve_by_constraint, list)
                            and len(self.reference_tradeoff_curve_by_constraint) > 0
                        )
                        or (isinstance(self.reference_cs_anchor_curve, list) and len(self.reference_cs_anchor_curve) > 0)
                        or (
                            isinstance(self.reference_constraint_anchor_curve, list)
                            and len(self.reference_constraint_anchor_curve) > 0
                        )
                    )
                if has_curve_in_memory:
                    self.phase = "reference_done"
                    self.phase_label = "Reference refreshed"
                    self.last_line = "Reference cache refreshed by reusing active chat runtime."
                    return "Reference cache refreshed (reused active chat runtime/model)."
                # No reusable cache yet: automatically stop active chat runtime and
                # run profiling path, then restart chat runtime below if configured.
                self.last_line = (
                    "No reusable reference cache. Temporarily stopping chat runtime for Profile LLM..."
                )
                self.last_error = ""
                await self._stop_chat_locked()
                self.chat_ready = False
            if detailed_profile and chat_alive_same_runtime:
                self.phase = "reference_refresh"
                self.phase_label = "Refreshing reference..."
                self.last_line = "Detailed Profile LLM requested. Restarting chat runtime for re-profiling..."
                self.last_error = ""
                await self._stop_chat_locked()
                self.chat_ready = False
            cmd = self._build_base_command(settings)
            # Force reference path execution with minimal probe run.
            cmd.extend([
                "--limit",
                "1",
                "--reference-only-exit-after-cache",
                # Profile LLM .
                "--deterministic",
            ])
            if detailed_profile:
                cmd.extend([
                    "--reference-force-refresh",
                    "--target-profile-force-refresh",
                    "--target-profile-node-list",
                    DETAILED_TARGET_PROFILE_NODE_LIST,
                ])
            else:
                cmd.extend([
                    "--target-profile-node-list",
                    DEFAULT_TARGET_PROFILE_NODE_LIST,
                ])
            self.phase = "reference_refresh"
            self.phase_label = "Refreshing reference..."
            if detailed_profile:
                self.last_line = "Refreshing reference cache with detailed (5-step) target profiling..."
            else:
                self.last_line = "Refreshing reference cache..."
            self.last_error = ""
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self.repo_root),
                env={**os.environ, "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python"},
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            assert proc.stdout is not None
            assert proc.stderr is not None
            out_task = asyncio.create_task(self._stream_reader(proc.stdout, is_stderr=False))
            err_task = asyncio.create_task(self._stream_reader(proc.stderr, is_stderr=True))
            await proc.wait()
            await asyncio.gather(out_task, err_task, return_exceptions=True)
            self._load_tradeoff_from_latest_reference(settings)
            if proc.returncode == 0:
                self.phase = "reference_done"
                self.phase_label = "Reference refreshed"
                keep_after_profile = str(
                    os.environ.get("AUTODRAFT_KEEP_TARGET_CONNECTION_AFTER_PROFILE", "1")
                ).strip().lower() not in {"0", "false", "no", "off"}
                mode_is_chat = str(settings.get("mode", "chat")).lower() == "chat"
                if (
                    keep_after_profile
                    and mode_is_chat
                    and (not self._is_server_only_api_selected(settings))
                ):
                    new_key = self._settings_key(settings)
                    chat_alive_same_runtime = (
                        self._chat_proc is not None
                        and self._chat_proc.returncode is None
                        and self._chat_settings_key == new_key
                    )
                    if chat_alive_same_runtime:
                        self.chat_ready = True
                        self.phase = "chat_ready"
                        self.phase_label = "Chat ready"
                        self.last_line = "Reference refreshed and existing chat runtime kept alive."
                        return "Reference cache refreshed. Chat runtime kept alive."
                    await self._stop_chat_locked()
                    self.phase = "chat_booting"
                    self.phase_label = "Booting chat runtime..."
                    self.last_line = "Reference refreshed. Starting persistent chat runtime..."
                    self.chat_ready = False
                    cmd = self._build_persistent_chat_command(settings)
                    self._chat_proc = await asyncio.create_subprocess_exec(
                        *cmd,
                        cwd=str(self.repo_root),
                        env={**os.environ, "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python"},
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    assert self._chat_proc.stdout is not None
                    assert self._chat_proc.stderr is not None
                    self._chat_stdout_task = asyncio.create_task(
                        self._chat_stream_reader(self._chat_proc.stdout, is_stderr=False)
                    )
                    self._chat_stderr_task = asyncio.create_task(
                        self._chat_stream_reader(self._chat_proc.stderr, is_stderr=True)
                    )
                    self._chat_settings_key = new_key
                    ok_ready, ready_msg = await self._wait_chat_ready(
                        timeout_s=float(os.environ.get("AUTODRAFT_CHAT_READY_TIMEOUT_SEC", "180.0"))
                    )
                    if not ok_ready:
                        self.phase = "chat_error"
                        self.phase_label = "Chat startup failed"
                        self.last_error = ready_msg
                        await self._stop_chat_locked()
                        return (
                            "Reference cache refreshed, but failed to keep chat runtime alive: "
                            f"{ready_msg}"
                        )
                    self.phase = "chat_ready"
                    self.phase_label = "Chat ready"
                    self.last_line = "Reference refreshed. Chat runtime is ready and connected."
                    return "Reference cache refreshed. Chat runtime is ready."
                return "Reference cache refreshed."
            self.phase = "error"
            self.phase_label = "Reference refresh failed"
            return f"Reference refresh failed (exit_code={proc.returncode})."

    async def start_chat_session(self, settings: Dict) -> str:
        async with self._lock:
            self.current_settings = dict(settings)
            err = self._validate_execution_server(settings)
            if err:
                return err
            if self.running:
                return "Benchmark is running. Stop benchmark first."
            if self._is_server_only_api_selected(settings):
                await self._stop_chat_locked()
                self._chat_settings_key = self._settings_key(settings)
                self.chat_ready = True
                self.phase = "chat_ready"
                self.phase_label = "Chat ready"
                self.last_line = "Server-only API chat is ready."
                return "Server-only API chat ready."
            new_key = self._settings_key(settings)
            if self._chat_proc is not None and self._chat_proc.returncode is None and self._chat_settings_key == new_key:
                return "Chat session already running."
            await self._stop_chat_locked()
            # Auto-prime reference cache on first Start when no reusable curve exists.
            has_reference_curve = bool(
                (isinstance(self.reference_tradeoff_curve_cs0_1, list) and len(self.reference_tradeoff_curve_cs0_1) > 0)
                or (isinstance(self.reference_tradeoff_curve_by_constraint, list) and len(self.reference_tradeoff_curve_by_constraint) > 0)
                or (isinstance(self.reference_cs_anchor_curve, list) and len(self.reference_cs_anchor_curve) > 0)
                or (
                    isinstance(self.reference_constraint_anchor_curve, list)
                    and len(self.reference_constraint_anchor_curve) > 0
                )
            )
            if not has_reference_curve:
                self._load_tradeoff_from_latest_reference(settings)
                has_reference_curve = bool(
                    (isinstance(self.reference_tradeoff_curve_cs0_1, list) and len(self.reference_tradeoff_curve_cs0_1) > 0)
                    or (isinstance(self.reference_tradeoff_curve_by_constraint, list) and len(self.reference_tradeoff_curve_by_constraint) > 0)
                    or (isinstance(self.reference_cs_anchor_curve, list) and len(self.reference_cs_anchor_curve) > 0)
                    or (
                        isinstance(self.reference_constraint_anchor_curve, list)
                        and len(self.reference_constraint_anchor_curve) > 0
                    )
                )
            if not has_reference_curve:
                self.phase = "reference_refresh"
                self.phase_label = "Refreshing reference..."
                self.last_line = "No reference cache found. Running Profile LLM once before Start..."
                self.last_error = ""
                ref_cmd = self._build_base_command(settings)
                ref_cmd.extend([
                    "--limit",
                    "1",
                    "--reference-only-exit-after-cache",
                    "--deterministic",
                ])
                ref_proc = await asyncio.create_subprocess_exec(
                    *ref_cmd,
                    cwd=str(self.repo_root),
                    env={**os.environ, "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python"},
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                assert ref_proc.stdout is not None
                assert ref_proc.stderr is not None
                ref_out_task = asyncio.create_task(self._stream_reader(ref_proc.stdout, is_stderr=False))
                ref_err_task = asyncio.create_task(self._stream_reader(ref_proc.stderr, is_stderr=True))
                await ref_proc.wait()
                await asyncio.gather(ref_out_task, ref_err_task, return_exceptions=True)
                self._load_tradeoff_from_latest_reference(settings)
                if ref_proc.returncode != 0:
                    self.phase = "error"
                    self.phase_label = "Reference refresh failed"
                    return f"Failed to start chat session: automatic reference refresh failed (exit_code={ref_proc.returncode})."
            self.phase = "draft_loading"
            self.phase_label = "Loading draft runtime..."
            self.chat_ready = False
            cmd = self._build_persistent_chat_command(settings)
            self._chat_proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self.repo_root),
                env={**os.environ, "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python"},
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            assert self._chat_proc.stdout is not None
            assert self._chat_proc.stderr is not None
            self._chat_stdout_task = asyncio.create_task(self._chat_stream_reader(self._chat_proc.stdout, is_stderr=False))
            self._chat_stderr_task = asyncio.create_task(self._chat_stream_reader(self._chat_proc.stderr, is_stderr=True))
            self._chat_settings_key = new_key
            self.phase = "chat_booting"
            self.phase_label = "Booting chat runtime..."
            self.last_line = "Waiting for chat runtime ready signal..."
            ok_ready, ready_msg = await self._wait_chat_ready(
                timeout_s=float(os.environ.get("AUTODRAFT_CHAT_READY_TIMEOUT_SEC", "180.0"))
            )
            if not ok_ready:
                self.phase = "chat_error"
                self.phase_label = "Chat startup failed"
                self.last_error = ready_msg
                await self._stop_chat_locked()
                return f"Failed to start chat session: {ready_msg}"
            self.phase = "chat_ready"
            self.phase_label = "Chat ready"
            return "Chat session ready."

    async def _stop_chat_locked(self):
        if self._chat_proc is None:
            return
        if self._chat_proc.returncode is None:
            try:
                if self._chat_proc.stdin is not None:
                    async with self._chat_stdin_lock:
                        self._chat_proc.stdin.write((json.dumps({"cmd": "stop"}) + "\n").encode("utf-8"))
                        await self._chat_proc.stdin.drain()
            except Exception:
                pass
            try:
                self._chat_proc.terminate()
            except Exception:
                pass
            try:
                await asyncio.wait_for(self._chat_proc.wait(), timeout=2.0)
            except Exception:
                pass
        self._chat_proc = None
        self._chat_settings_key = None
        self.chat_ready = False
        self.phase = "stopped"
        self.phase_label = "Stopped"

    async def _wait_chat_ready(self, timeout_s: float = 180.0) -> tuple[bool, str]:
        loop = asyncio.get_event_loop()
        normal_budget_s = max(1.0, float(timeout_s))
        profile_wait_budget_s = max(
            1.0,
            float(os.environ.get("AUTODRAFT_CHAT_PROFILE_WAIT_TIMEOUT_SEC", "7200.0")),
        )
        normal_elapsed_s = 0.0
        profile_wait_elapsed_s = 0.0
        last_tick = loop.time()
        while True:
            if self.chat_ready:
                return True, "ready"
            proc = self._chat_proc
            if proc is None:
                return False, "chat process is not available."
            if proc.returncode is not None:
                return False, f"chat process exited early (exit_code={proc.returncode})."
            if self.phase == "chat_error":
                return False, self.last_error or "chat runtime reported startup error."
            now = loop.time()
            dt = max(0.0, now - last_tick)
            last_tick = now
            if self.phase == "target_profiling_wait":
                profile_wait_elapsed_s += dt
                if profile_wait_elapsed_s >= profile_wait_budget_s:
                    return (
                        False,
                        "chat runtime ready timeout while waiting for target profiling "
                        f"({int(profile_wait_budget_s)}s).",
                    )
            else:
                normal_elapsed_s += dt
                if normal_elapsed_s >= normal_budget_s:
                    return False, "chat runtime ready timeout."
            await asyncio.sleep(0.2)

    async def stop_chat_session(self, settings: Optional[Dict] = None) -> str:
        async with self._lock:
            await self._stop_chat_locked()
            msg = "Chat session stopped. Draft runtime unloaded."
            if settings and not self._is_server_only_api_selected(settings):
                unload = await self.unload_remote_target_model(settings)
                if unload.get("ok"):
                    msg = f"{msg} Target model unloaded."
                else:
                    msg = f"{msg} {unload.get('message', '')}".strip()
            self._reset_tradeoff_state()
            return msg

    async def shutdown_target(self, settings: Dict) -> str:
        async with self._lock:
            if self.running:
                if self._proc is not None:
                    self._proc.terminate()
                elif self._run_task is not None and not self._run_task.done():
                    self._run_task.cancel()
            await self._stop_chat_locked()
            result = await self.shutdown_remote_target(settings)
            return str(result.get("message", "Target shutdown request finished."))

    async def update_chat_runtime_settings(self, settings: Dict):
        """Push live settings to persistent chat runtime (best effort)."""
        proc = self._chat_proc
        if proc is None or proc.returncode is not None or proc.stdin is None:
            return
        payload = {
            "cmd": "set",
            "cost_sensitivity": float(settings.get("cost", 0.15)),
            "max_new_tokens": int(settings.get("max_new_tokens", 512)),
            "proactive_drafting": bool(settings.get("proactive_drafting", False)),
            "constraint_target": str(settings.get("constraint_target", "metric")),
            "metric_constraint_per_1m_token": float(
                settings.get("metric_constraint_per_1m_token", 14.0)
            ),
            "min_tps_constraint": float(settings.get("min_tps_constraint", 0.0) or 0.0),
        }
        try:
            async with self._chat_stdin_lock:
                proc.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
                await proc.stdin.drain()
        except Exception:
            return

    def list_server_candidates(self) -> List[Dict]:
        return self.registry.list_servers()

    async def add_server_candidate(self, payload: Dict) -> Dict:
        async with self._lock:
            name = str(payload.get("name", "User Server")).strip() or "User Server"
            host = str(payload.get("host", "")).strip()
            endpoint = str(payload.get("endpoint", "")).strip()
            port_raw = payload.get("port")
            try:
                port_val = int(port_raw) if port_raw is not None and str(port_raw).strip() else None
            except Exception:
                port_val = None

            if host:
                if port_val is None:
                    try:
                        port_val = int(os.environ.get("AUTODRAFT_PORT", "26001"))
                    except Exception:
                        port_val = 26001
                endpoint = f"{host}:{int(port_val)}"

            base_model = str(
                os.environ.get("AUTODRAFT_BASE_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
            ).strip() or "meta-llama/Llama-3.3-70B-Instruct"
            models = payload.get("models") if isinstance(payload.get("models"), list) else []
            if not models:
                catalog_local = self.registry.get("catalog_local_target")
                if catalog_local is not None and getattr(catalog_local, "models", None):
                    models = [
                        {"model_id": str(m.model_id), "label": str(m.label or m.model_id)}
                        for m in catalog_local.models
                        if str(getattr(m, "model_id", "")).strip()
                    ]
                if not models:
                    models = [
                        {"model_id": base_model, "label": f"{base_model} (default)"},
                        {"model_id": "meta-llama/Llama-3.3-70B-Instruct", "label": "Meta Llama 3.3 70B Instruct"},
                        {"model_id": "Qwen/Qwen2.5-14B-Instruct", "label": "Qwen2.5 14B Instruct"},
                        {"model_id": "Qwen/Qwen2.5-32B-Instruct", "label": "Qwen2.5 32B Instruct"},
                        {"model_id": "Qwen/Qwen3-14B", "label": "Qwen3 14B"},
                        {"model_id": "Qwen/Qwen3-32B", "label": "Qwen3 32B"},
                    ]

            metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
            metadata = dict(metadata)
            metadata.setdefault("server_name", name)
            spec = self.registry.add_user_server(
                name=name,
                endpoint=endpoint,
                protocol="autodraft_target",
                server_type="external",
                requires_api_key=False,
                models=models,
                default_model_id=str(payload.get("default_model_id") or "") or base_model,
                api_key=None,
                metadata=metadata,
            )
            self._persist_server_env(spec)
            return {"ok": True, "server": spec.to_dict(include_secret=False)}

    async def update_server_api_key(self, server_id: str, api_key: str) -> Dict:
        async with self._lock:
            ok = self.registry.set_api_key(server_id, api_key)
            return {"ok": ok}

    async def remove_server_candidate(self, server_id: str) -> Dict:
        async with self._lock:
            ok = self.registry.remove_server(server_id)
            return {"ok": ok}

    async def run_probing(self, settings: Dict, payload: Optional[Dict] = None) -> Dict:
        payload = payload or {}
        selected = payload.get("server_ids")
        model_overrides = payload.get("model_overrides") if isinstance(payload.get("model_overrides"), dict) else {}
        objective_mode = str(settings.get("objective_selection_mode", "blend")).lower()
        objective_mode = "constraint" if objective_mode == "constraint" else "blend"
        curve_mode = "constraint" if objective_mode == "constraint" else "cost_sensitivity"
        metric_preference = str(payload.get("metric_preference") or self.metric_preference or "total_cost")
        self.metric_preference = metric_preference

        servers = self.registry.list_servers()
        selected_set = set(selected) if isinstance(selected, list) else None

        if objective_mode == "constraint":
            constraint_target = str(settings.get("constraint_target", "metric")).lower()
            if constraint_target == "tps":
                center = float(settings.get("min_tps_constraint", 0.0) or 0.0)
                if center <= 0:
                    center = float(self.stats.get("throughput", 0.0) or 0.0)
                if center <= 0:
                    center = 10.0
                low = max(0.1, center * 0.7)
                high = max(low + 0.1, center * 1.3)
                curve_selectors = [low, (low + center) / 2.0, center, (high + center) / 2.0, high]
            else:
                center = float(settings.get("metric_constraint_per_1m_token", 14.0))
                low = max(0.1, center * 0.7)
                high = max(low + 0.1, center * 1.3)
                curve_selectors = [low, (low + center) / 2.0, center, (high + center) / 2.0, high]
        else:
            curve_selectors = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        curve_selectors = [float(v) for v in curve_selectors]

        request = ProbeRequest(
            prompt=str(payload.get("prompt", "Summarize speculative decoding in one sentence.")),
            max_tokens=int(payload.get("max_tokens", 64)),
            temperature=float(payload.get("temperature", 0.0)),
            timeout_s=float(payload.get("timeout_s", 25.0)),
            warmup_runs=int(payload.get("warmup_runs", 1)),
            measured_runs=int(payload.get("measured_runs", 1)),
        )

        rows = []
        curves = []
        self.probe_curve_mode = curve_mode
        self.probe_status = {"running": True, "last_error": "", "updated_at": time.time()}
        for row in servers:
            server_id = str(row.get("server_id"))
            if selected_set is not None and server_id not in selected_set:
                continue
            spec = self.registry.get(server_id)
            if spec is None or not spec.enabled:
                continue
            model_id = model_overrides.get(server_id) or spec.default_model_id
            models = [m for m in spec.models if m.model_id == model_id] if model_id else []
            if not models and spec.models:
                models = [spec.models[0]]
            if not models:
                rows.append({"ok": False, "server_id": server_id, "error": "no_models"})
                continue
            model = models[0]
            curve_points = []
            curve_source = "probe"
            candidate_settings = dict(settings)
            candidate_settings["selected_server_id"] = str(server_id)
            candidate_settings["selected_model_id"] = str(model.model_id)
            runtime_key = self._settings_key(candidate_settings)
            # Priority 1: active chat/runtime memory (already loaded model + draft runtime).
            if (
                self._chat_proc is not None
                and self._chat_proc.returncode is None
                and self.chat_ready
                and self._chat_settings_key == runtime_key
            ):
                curve_points = self._curve_points_from_runtime_state(objective_mode)
                if curve_points:
                    curve_source = "active_runtime"
            # Priority 2: reference cache on disk (Profile LLM result).
            if not curve_points:
                cached_payload = self._read_latest_reference_payload(candidate_settings)
                if isinstance(cached_payload, dict):
                    cached_payload = {
                        **cached_payload,
                        "constraint_target": candidate_settings.get("constraint_target", "metric"),
                    }
                curve_points = self._curve_points_from_reference_payload(cached_payload, objective_mode)
                if curve_points:
                    curve_source = "reference_cache"
            # Priority 3: no reference available -> run short probe sweep.
            if not curve_points:
                for selector_val in curve_selectors:
                    result = await asyncio.to_thread(
                        self.probe_runner.probe_server_model, spec, model, request
                    )
                    if result.get("ok"):
                        curve_points.append(
                            {
                                "selector_value": float(selector_val),
                                "metric_per_1m": float(result.get("metric_per_1m"))
                                if result.get("metric_per_1m") is not None
                                else None,
                                "throughput_tps": float(result.get("throughput_tps"))
                                if result.get("throughput_tps") is not None
                                else None,
                            }
                        )
            if curve_points:
                curve_points = sorted(
                    curve_points,
                    key=lambda p: float(p.get("selector_value", 0.0)),
                )
                curves.append(
                    {
                        "server_id": str(server_id),
                        "server_name": str(spec.name),
                        "model_id": str(model.model_id),
                        "curve_mode": curve_mode,
                        "source": curve_source,
                        "points": curve_points,
                    }
                )
                # Recommendation is based on current selector on each server curve.
                if objective_mode == "constraint":
                    if str(settings.get("constraint_target", "metric")).lower() == "tps":
                        current_selector = float(settings.get("min_tps_constraint", 0.0) or 0.0)
                    else:
                        current_selector = float(settings.get("metric_constraint_per_1m_token", 14.0))
                else:
                    current_selector = float(settings.get("cost", 0.15))
                nearest = min(curve_points, key=lambda p: abs(float(p.get("selector_value", 0.0)) - current_selector))
                rows.append(
                    {
                        "ok": True,
                        "server_id": str(server_id),
                        "server_name": str(spec.name),
                        "model_id": str(model.model_id),
                        "model_label": str(model.label),
                        "throughput_tps": nearest.get("throughput_tps"),
                        "metric_per_1m": nearest.get("metric_per_1m"),
                        "probe_mode": "server_curve_profile",
                        "profile_source": curve_source,
                    }
                )
            else:
                rows.append({"ok": False, "server_id": server_id, "model_id": str(model.model_id), "error": "curve_probe_failed"})

        scored, summary = build_recommendations(rows, objective_mode, metric_preference)
        self.probe_rows = scored if scored else rows
        self.probe_curves = curves
        self.recommendation_summary = summary
        self.probe_status = {"running": False, "last_error": "", "updated_at": time.time()}
        return {
            "ok": True,
            "rows": self.probe_rows,
            "summary": self.recommendation_summary,
            "curves": curves,
            "curve_mode": curve_mode,
        }

    async def update_recommendations(self, settings: Dict, metric_preference: str) -> Dict:
        objective_mode = str(settings.get("objective_selection_mode", "blend")).lower()
        objective_mode = "constraint" if objective_mode == "constraint" else "blend"
        self.metric_preference = metric_preference or self.metric_preference
        scored, summary = build_recommendations(
            list(self.probe_rows), objective_mode, self.metric_preference
        )
        self.probe_rows = scored
        self.recommendation_summary = summary
        return {"ok": True, "rows": self.probe_rows, "summary": self.recommendation_summary}

    async def generate_chat_reply(
        self,
        user_text: str,
        settings: Dict,
        on_partial: Optional[Callable[[Dict], Awaitable[None]]] = None,
    ) -> Dict:
        """Generate one turn via persistent chat-mode process."""
        async with self._lock:
            self.current_settings = dict(settings)
            err = self._validate_execution_server(settings)
            if err:
                return {"reply": err, "token_trace": None}
            if self.running:
                return {"reply": "Benchmark is running. Stop it first before chat generation.", "token_trace": None}
            if self._is_server_only_api_selected(settings):
                self.phase = "chat_generating"
                self.phase_label = "Generating reply..."
                try:
                    out = await asyncio.to_thread(self._api_server_only_chat_sync, user_text, settings)
                    if not out.get("ok"):
                        self.phase = "chat_error"
                        self.phase_label = "Chat error"
                        return {"reply": f"Server-only API error: {out.get('error', 'unknown_error')}", "token_trace": None}
                    final_stats = out.get("stats") if isinstance(out.get("stats"), dict) else None
                    if final_stats:
                        self.stats["gpu_energy"] = float(final_stats.get("gpu_energy", self.stats["gpu_energy"]))
                        self.stats["draft_cost"] = float(final_stats.get("draft_cost", self.stats["draft_cost"]))
                        self.stats["target_cost"] = float(final_stats.get("target_cost", self.stats["target_cost"]))
                        self.stats["throughput"] = float(final_stats.get("throughput", self.stats["throughput"]))
                        self.stats["cost_per_1m"] = float(final_stats.get("cost_per_1m", self.stats["cost_per_1m"]))
                        self.stats["draft_energy_per_1m"] = float(
                            final_stats.get("draft_energy_per_1m", self.stats["draft_energy_per_1m"])
                        )
                    self.phase = "chat_ready"
                    self.phase_label = "Chat ready"
                    self.chat_ready = True
                    return {
                        "reply": str(out.get("reply", "")),
                        "token_trace": None,
                        "final_stats": final_stats,
                    }
                except urllib.error.HTTPError as http_err:
                    detail = ""
                    try:
                        detail = http_err.read().decode("utf-8", errors="ignore")
                    except Exception:
                        detail = str(http_err)
                    self.phase = "chat_error"
                    self.phase_label = "Chat error"
                    self.last_error = f"http_error:{http_err.code}:{detail[:220]}"
                    return {"reply": f"Server-only API HTTP error ({http_err.code}).", "token_trace": None}
                except Exception as exc:
                    self.phase = "chat_error"
                    self.phase_label = "Chat error"
                    self.last_error = str(exc)
                    return {"reply": f"Server-only API error: {exc}", "token_trace": None}
            # restart session if settings changed
            new_key = self._settings_key(settings)
            if self._chat_proc is None or self._chat_proc.returncode is not None or self._chat_settings_key != new_key:
                preload = await self.ensure_remote_target_loaded(settings)
                if not preload.get("ok"):
                    return {"reply": str(preload.get("message", "Failed to prepare target runtime.")), "token_trace": None}
                await self._stop_chat_locked()
                cmd = self._build_persistent_chat_command(settings)
                self._chat_proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=str(self.repo_root),
                    env={**os.environ, "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python"},
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                assert self._chat_proc.stdout is not None
                assert self._chat_proc.stderr is not None
                self._chat_stdout_task = asyncio.create_task(self._chat_stream_reader(self._chat_proc.stdout, is_stderr=False))
                self._chat_stderr_task = asyncio.create_task(self._chat_stream_reader(self._chat_proc.stderr, is_stderr=True))
                self._chat_settings_key = new_key
                self.phase = "chat_booting"
                self.phase_label = "Booting chat runtime..."
                self.chat_ready = False

            if self._chat_proc is None or self._chat_proc.stdin is None:
                self.phase = "chat_error"
                self.phase_label = "Chat start failed"
                return {"reply": "Failed to start persistent chat session.", "token_trace": None}
            if self._chat_proc.returncode is not None:
                self.phase = "chat_error"
                self.phase_label = "Chat process exited"
                return {
                    "reply": f"Chat process exited early (exit_code={self._chat_proc.returncode}). {self.last_error}".strip(),
                    "token_trace": None,
                }
            try:
                while not self._chat_queue.empty():
                    self._chat_queue.get_nowait()
            except Exception:
                pass
            self.phase = "chat_generating"
            self.phase_label = "Generating reply..."
            req = {
                "cmd": "chat",
                "text": user_text,
                "cost_sensitivity": float(settings.get("cost", 0.15)),
                "max_new_tokens": int(settings.get("max_new_tokens", 512)),
                "constraint_target": str(settings.get("constraint_target", "metric")),
                "metric_constraint_per_1m_token": float(settings.get("metric_constraint_per_1m_token", 14.0)),
                "min_tps_constraint": float(settings.get("min_tps_constraint", 0.0) or 0.0),
                "proactive_drafting": bool(settings.get("proactive_drafting", False)),
            }
            async with self._chat_stdin_lock:
                self._chat_proc.stdin.write((json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8"))
                await self._chat_proc.stdin.drain()
            last_partial_stats = None
            try:
                while True:
                    msg = await asyncio.wait_for(self._chat_queue.get(), timeout=300.0)
                    if msg.get("type") == "chat_partial":
                        if isinstance(msg.get("stats"), dict):
                            last_partial_stats = msg.get("stats")
                        if on_partial is not None:
                            try:
                                await on_partial(msg)
                            except Exception:
                                pass
                        continue
                    if msg.get("type") == "chat_reply":
                        self.phase = "chat_ready"
                        self.phase_label = "Chat ready"
                        reply_final_stats = (
                            msg.get("final_stats")
                            if isinstance(msg.get("final_stats"), dict)
                            else last_partial_stats
                        )
                        return {
                            "reply": str(msg.get("reply", "")),
                            "token_trace": msg.get("token_trace", None),
                            "final_stats": reply_final_stats,
                        }
                    if msg.get("type") == "error":
                        self.phase = "chat_error"
                        self.phase_label = "Chat error"
                        return {"reply": f"Hybrid decoding error: {msg.get('message', '')}", "token_trace": None}
            except asyncio.TimeoutError:
                self.phase = "chat_error"
                self.phase_label = "Chat timeout"
                return {"reply": "Hybrid decoding timed out.", "token_trace": None}

    def snapshot(self) -> Dict:
        out = dict(self.stats)
        chat_running = bool(self._chat_proc is not None and self._chat_proc.returncode is None)
        chat_exit_code = (
            self._chat_proc.returncode
            if self._chat_proc is not None and self._chat_proc.returncode is not None
            else None
        )
        out.update(
            {
                "last_line": self.last_line,
                "last_error": self.last_error,
                "running": self.running,
                "chat_running": chat_running,
                "chat_ready": bool(self.chat_ready),
                "chat_exit_code": chat_exit_code,
                "exit_code": self.exit_code,
                "result_json": self.last_result_json,
                "phase": self.phase,
                "phase_label": self.phase_label,
                "constraint_target": self.current_settings.get("constraint_target", "metric"),
                "benchmark_dataset": self.current_settings.get("benchmark_dataset", "mt_bench"),
                "feasible_constraint_range_per_1m": self.feasible_constraint_range_per_1m,
                "feasible_tps_range": self.feasible_tps_range,
                "reference_tradeoff_curve_cs0_1": self.reference_tradeoff_curve_cs0_1,
                "reference_tradeoff_curve_by_constraint": self.reference_tradeoff_curve_by_constraint,
                "reference_cs_anchor_curve": self.reference_cs_anchor_curve,
                "reference_constraint_anchor_curve": self.reference_constraint_anchor_curve,
                "server_candidates": self.registry.list_servers(),
                "probe_status": self.probe_status,
                "server_probe_rows": self.probe_rows,
                "server_probe_curves": self.probe_curves,
                "server_probe_curve_mode": self.probe_curve_mode,
                "recommendations": self.recommendation_summary,
                "metric_preference": self.metric_preference,
            }
        )
        return out

    async def load_latest_result_metrics(self) -> Dict:
        """Optional helper: read final metrics after run completion."""
        p = self.last_result_json
        if not p:
            return {}
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            gs = data.get("generation_stats", {})
            return {
                "throughput": float(gs.get("token_per_second", 0.0) or 0.0),
                "cost_per_1m": float(gs.get("cost_per_1m_tokens", 0.0) or 0.0),
                "draft_energy_per_1m": float(gs.get("draft_energy_kwh_per_1m_tokens", 0.0) or 0.0),
            }
        except Exception:
            return {}

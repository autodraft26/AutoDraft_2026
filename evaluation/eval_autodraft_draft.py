import argparse
import copy
import glob
import gc
import hashlib
import json
import os
import random
import select
import socket
import subprocess
import tempfile
import time
import threading
import queue
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import shortuuid
import torch
from fastchat.llm_judge.common import load_questions
from fastchat.model import get_conversation_template
from transformers import AutoTokenizer
from tqdm import tqdm
import importlib
from opt_classic.utils import (
    CPUPowerMonitor,
    GPUMonitor,
    prepare_logits_processor,
    recv_json_with_size,
    send_json_with_size,
)
from opt_classic.modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from opt_classic.tree import Tree
import numpy as np
np.set_printoptions(threshold=np.inf)

# Keep draft profiling widths aligned with Tree candidate widths.
DRAFT_TREE_PROFILE_WIDTHS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
PROFILE_WARMUP_RUNS = 2
PROFILE_BURNIN_RUNS = 1
ONLINE_PROFILE_LR_DEFAULT = 0.05


def _normalize_objective_metric(value: Optional[str]) -> str:
    metric = str(value).lower() if value is not None else "total_cost"
    if metric == "cost":
        return "total_cost"
    return metric


class _AsyncRecvHandle:
    """Thread-like handle for one recv task submitted to a reusable recv worker."""

    def __init__(self):
        self._done = threading.Event()
        self._payload = None
        self._error = None

    def is_alive(self):
        return not self._done.is_set()

    def join(self, timeout=None):
        self._done.wait(timeout)

    def set_result(self, payload):
        self._payload = payload
        self._done.set()

    def set_error(self, error):
        self._error = error
        self._done.set()

    def get(self):
        self._done.wait()
        if self._error is not None:
            raise self._error
        reply, t2d_bytes, recv_end_time = self._payload
        return reply, int(t2d_bytes), float(recv_end_time)


class _ReusableRecvWorker:
    """Reuse one recv thread instead of creating a new thread every decoding step."""

    def __init__(self):
        self._queue = queue.Queue()
        self._shutdown = object()
        self.submitted = 0
        self.completed = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, sock: socket.socket):
        handle = _AsyncRecvHandle()
        self.submitted += 1
        self._queue.put((sock, handle))
        return handle

    def _run(self):
        while True:
            item = self._queue.get()
            if item is self._shutdown:
                self._queue.task_done()
                break
            sock, handle = item
            try:
                reply, t2d_bytes = recv_json_with_size(sock)
                handle.set_result((reply, t2d_bytes, time.time()))
            except Exception as e:
                handle.set_error(e)
            finally:
                self.completed += 1
                self._queue.task_done()

    def shutdown(self):
        self._queue.put(self._shutdown)
        self._thread.join()


def _start_recv_json_async(sock: socket.socket, recv_worker: _ReusableRecvWorker = None):
    """
    Run socket recv in a dedicated thread.
    The return queue contains one of the following:
      - ("ok", (reply, bytes_recv, recv_end_time))
      - ("err", exception)
    """
    if recv_worker is not None:
        return recv_worker.submit(sock), None

    out_q: "queue.Queue[Tuple[str, Any]]" = queue.Queue(maxsize=1)

    def _worker() -> None:
        try:
            reply, t2d_bytes = recv_json_with_size(sock)
            out_q.put(("ok", (reply, t2d_bytes, time.time())))
        except Exception as e:
            out_q.put(("err", e))

    th = threading.Thread(target=_worker, daemon=True)
    th.start()
    return th, out_q


def _await_recv_json_async(
    recv_thread,
    out_q: "queue.Queue[Tuple[str, Any]]",
) -> Tuple[dict, int, float]:
    if hasattr(recv_thread, "get"):
        return recv_thread.get()
    tag, payload = out_q.get()
    recv_thread.join()
    if tag == "err":
        raise payload
    reply, t2d_bytes, recv_end_time = payload
    return reply, int(t2d_bytes), float(recv_end_time)


try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def set_seed(seed: int):
    """Fixed torch, random, and numpy seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def set_deterministic():
    """Disable PyTorch non-deterministic algorithm for identical results even on GPU operations (some operations may be slower)."""
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def get_kv_llama_class(base_model_path: str):

    model_path_l = (base_model_path or "").lower()
    is_llama3_family = ("llama-3" in model_path_l) or ("llama3" in model_path_l)
    is_qwen3_family = ("qwen3" in model_path_l) or ("qwen-3" in model_path_l) or ("qwen/qwen3" in model_path_l)
    is_qwen2_family = (not is_qwen3_family) and (
        ("qwen2" in model_path_l) or ("qwen-2" in model_path_l) or ("qwen/qwen2" in model_path_l)
    )

    if is_qwen3_family:
        module_name = "opt_classic.modeling_qwen3_kv"
        class_name = "Qwen3ForCausalLM"
    elif is_qwen2_family:
        module_name = "opt_classic.modeling_qwen2_kv"
        class_name = "Qwen2ForCausalLM"
    elif is_llama3_family:
        module_name = "opt_classic.modeling_llama3_kv"
        class_name = "LlamaForCausalLM"
    else:
        module_name = "opt_classic.modeling_llama_kv"
        class_name = "LlamaForCausalLM"

    print(f'importing {module_name}.{class_name}')
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, class_name)
    except Exception as e:
        raise RuntimeError(f"Failed to import {module_name}.{class_name}: {e}")
    

def _resolve_data_root(parent_dir: str) -> str:
    """Resolve the directory used for ``data/profile`` and ``data/reference``
    caches. ``AUTODRAFT_DATA_DIR`` takes precedence over the source-checkout
    default (``<repo_root>/data``) so PyPI-installed users don't end up
    writing into their site-packages tree."""
    override = os.environ.get("AUTODRAFT_DATA_DIR")
    if override:
        return override
    return os.path.join(parent_dir, "data")


def _normalize_remote_model_id(value: str) -> str:
    return str(value or "").strip().lower()


def _normalize_quantization_mode(value: str, default: str = "8bit") -> str:
    q = str(value or default).strip().lower()
    if q in {"none", "4bit", "8bit"}:
        return q
    return str(default).strip().lower()


def _build_target_quantization_fallback_chain(
    requested_quantization: str,
    *,
    load_in_4bit: bool,
    load_in_8bit: bool,
) -> List[str]:
    default_quant = "none"
    if bool(load_in_8bit):
        default_quant = "8bit"
    elif bool(load_in_4bit):
        default_quant = "4bit"
    preferred = _normalize_quantization_mode(requested_quantization, default=default_quant)
    order = ["none", "8bit", "4bit"]
    start_idx = order.index(preferred) if preferred in order else 0
    return order[start_idx:]


def _is_memory_related_load_error(message: str) -> bool:
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


def _is_target_reload_memory_related_error(message: str) -> bool:
    return _is_memory_related_load_error(message)


def _build_draft_quantization_fallback_chain(
    requested_quantization: str,
    *,
    load_in_4bit: bool,
    load_in_8bit: bool,
) -> List[str]:
    default_quant = "none"
    if bool(load_in_8bit):
        default_quant = "8bit"
    elif bool(load_in_4bit):
        default_quant = "4bit"
    preferred = _normalize_quantization_mode(requested_quantization, default=default_quant)
    order = ["none", "8bit", "4bit"]
    start_idx = order.index(preferred) if preferred in order else 0
    return order[start_idx:]


def _build_quantization_config_for_mode(quant_mode: str):
    mode = _normalize_quantization_mode(quant_mode, default="none")
    if mode == "8bit":
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_skip_modules=["lm_head"],
        )
    if mode == "4bit":
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    return None


def _release_partial_draft_model(model_obj=None):
    try:
        if model_obj is not None:
            del model_obj
    except Exception:
        pass
    try:
        gc.collect()
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _draft_model_prefers_default_8bit(model_path: str) -> bool:
    model_l = str(model_path or "").strip().lower()
    if not model_l:
        return False
    # Keep 7B default as 8bit.
    return "7b" in model_l


def _draft_model_prefers_default_4bit(model_path: str) -> bool:
    model_l = str(model_path or "").strip().lower()
    if not model_l:
        return False
    # Updated policy: use 4bit by default for 8B draft models.
    return "8b" in model_l


def _target_model_prefers_default_8bit(model_path: str) -> bool:
    model_l = str(model_path or "").strip().lower()
    if not model_l:
        return False
    # Target default policy: 70B+/72B starts from 8bit, others from none.
    return ("70b" in model_l) or ("72b" in model_l)


DEFAULT_CHAT_SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  "
    "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
    "Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, "
    "or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, "
    "please don't share false information."
)


def _load_benchmark_questions(bench_name: str, question_file: Optional[str]) -> List[dict]:
    """Load benchmark questions for both local JSONL and HF datasets."""
    bench = str(bench_name or "").strip().lower().replace("_", "-")
    qf = str(question_file or "").strip()
    if qf:
        return load_questions(qf, None, None)
    script_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(script_dir)
    if bench in {"mt_bench", "mt-bench"}:
        return load_questions(f"{parent_dir}/data/mt_bench/question.jsonl", None, None)
    if bench == "gsm8k":
        return load_questions(f"{parent_dir}/data/gsm8k.jsonl", None, None)
    if bench == "humaneval":
        from datasets import load_dataset

        ds = load_dataset("openai_humaneval")["test"]
        return [{"question_id": ex["task_id"], "question": ex["prompt"]} for ex in ds]
    if bench == "ifeval":
        from datasets import load_dataset

        ds = load_dataset("google/IFEval")["train"]
        return [
            {
                "question_id": ex.get("key", idx),
                "question": ex["prompt"],
            }
            for idx, ex in enumerate(ds)
        ]
    if bench in {"math-500", "math500"}:
        from datasets import load_dataset

        ds = load_dataset("HuggingFaceH4/MATH-500")["test"]
        return [
            {
                "question_id": ex.get("unique_id") or ex.get("problem_id") or idx,
                "question": ex["problem"],
            }
            for idx, ex in enumerate(ds)
        ]
    if bench == "cnn_dailymail":
        from datasets import load_dataset

        ds = load_dataset("abisee/cnn_dailymail", "1.0.0")["test"]
        return [{"question_id": ex["id"], "question": ex["article"]} for ex in ds]
    raise ValueError(f"Undefined bench_name: {bench_name}")


def _extract_question_turns(question: dict, bench_name: str) -> List[str]:
    """Return a unified turns list for all supported benchmarks."""
    bench = str(bench_name or "").strip().lower().replace("_", "-")
    if bench in {"gsm8k", "humaneval", "cnn_dailymail", "ifeval", "math-500", "math500"}:
        q = question.get("question", "")
        return [q] if isinstance(q, str) and q else [str(q)]
    turns = question.get("turns", [])
    if isinstance(turns, list) and turns:
        return turns
    q = question.get("question", "")
    return [q] if isinstance(q, str) and q else [str(q)]


def _build_conversation_template_for_model(model_path: str):
    model_path_l = str(model_path or "").lower()
    if "vicuna" in model_path_l:
        return get_conversation_template("vicuna")
    if ("llama-3" in model_path_l) or ("llama3" in model_path_l):
        # FastChat template names vary by version; try common aliases.
        for template_name in ("llama-3", "llama3"):
            try:
                return get_conversation_template(template_name)
            except Exception:
                pass
    # Qwen2.5 / Qwen3 family uses ChatML format (<|im_start|>...<|im_end|>)
    if ("qwen" in model_path_l):
        try:
            return get_conversation_template("chatml")
        except Exception:
            pass
        # Fallback: manually construct ChatML-style conversation
        from fastchat.conversation import Conversation, SeparatorStyle
        conv = Conversation(
            name="qwen",
            system_template="<|im_start|>system\n{system_message}<|im_end|>",
            system_message="You are a helpful assistant.",
            roles=("<|im_start|>user", "<|im_start|>assistant"),
            sep_style=SeparatorStyle.CHATML,
            sep="<|im_end|>",
            stop_str="<|im_end|>",
        )
        return conv
    conv = get_conversation_template("llama-2-chat")
    conv.system_message = DEFAULT_CHAT_SYSTEM_PROMPT
    return conv


def _ensure_remote_target_model(
    sock: socket.socket,
    base_model_path: str,
    load_in_4bit: bool,
    load_in_8bit: bool,
    target_quantization: str = None,
    device_map: str = None,
    debug: bool = False,
) -> Dict[str, Any]:
    expected_model = str(base_model_path or "").strip()
    if not expected_model:
        raise RuntimeError("base_model_path is required to sync remote target model")
    # Model reload on target can take much longer than regular RPCs.
    reload_timeout_s = max(
        30.0,
        float(
            os.environ.get(
                "AUTODRAFT_TARGET_RELOAD_TIMEOUT_SEC",
                os.environ.get("AUTODRAFT_TARGET_RPC_TIMEOUT_SEC", "900.0"),
            )
        ),
    )
    prev_timeout = sock.gettimeout()
    if prev_timeout is None or float(prev_timeout) < reload_timeout_s:
        sock.settimeout(reload_timeout_s)
    fallback_chain = _build_target_quantization_fallback_chain(
        str(target_quantization or ""),
        load_in_4bit=bool(load_in_4bit),
        load_in_8bit=bool(load_in_8bit),
    )
    expected_quant = fallback_chain[0]
    quant_rank = {"none": 0, "8bit": 1, "4bit": 2}
    try:
        print("[Startup] target_sync_status", flush=True)
        send_json_with_size(sock, {"type": "status"})
        status_reply, _ = recv_json_with_size(sock)
        status_type = str(status_reply.get("type", ""))
        if status_type == "status_ok":
            loaded_model = str(status_reply.get("loaded_model", ""))
            loaded_quant = str(status_reply.get("quantization", "")).strip().lower()
            model_match = _normalize_remote_model_id(loaded_model) == _normalize_remote_model_id(expected_model)
            loaded_rank = quant_rank.get(str(loaded_quant), -1)
            expected_rank = quant_rank.get(str(expected_quant), -1)
            quant_match = (not loaded_quant) or (loaded_rank >= expected_rank)
            if model_match and quant_match:
                print("[Startup] target_already_loaded", flush=True)
                if debug:
                    print(f"[Target Sync] already loaded: model={loaded_model}, quant={loaded_quant or 'unknown'}")
                return {
                    "selected_quantization": str(loaded_quant or expected_quant),
                    "attempted_quantizations": [str(expected_quant)],
                    "fallback_applied": bool(loaded_quant and loaded_quant != expected_quant),
                    "status": "already_loaded",
                }
        elif status_type == "error":
            raise RuntimeError(f"target status failed: {status_reply.get('message', status_reply)}")
        else:
            raise RuntimeError(
                "target does not support status/reload protocol. "
                f"reply={status_reply}"
            )

        attempted_quants: List[str] = []
        failures: List[str] = []
        unload_cooldown_s = max(
            0.0, float(os.environ.get("AUTODRAFT_TARGET_RELOAD_UNLOAD_COOLDOWN_SEC", "0.25"))
        )
        for idx, quant_mode in enumerate(fallback_chain):
            attempted_quants.append(str(quant_mode))
            print(f"[Startup] target_reloading quant={quant_mode}", flush=True)
            try:
                send_json_with_size(sock, {"type": "unload_model"})
                unload_reply, _ = recv_json_with_size(sock)
                if str(unload_reply.get("type", "")) != "unload_ok" and debug:
                    print(f"[Target Sync][WARN] unload before fallback failed: {unload_reply}")
            except Exception as unload_exc:
                if debug:
                    print(f"[Target Sync][WARN] unload before fallback exception: {unload_exc}")
            if unload_cooldown_s > 0:
                time.sleep(unload_cooldown_s)
            send_json_with_size(
                sock,
                {
                    "type": "reload_model",
                    "base_model_path": expected_model,
                    "quantization": str(quant_mode),
                    "device_map": str(device_map or "auto"),
                },
            )
            reload_reply, _ = recv_json_with_size(sock)
            reload_type = str(reload_reply.get("type", ""))
            if reload_type == "reload_ok":
                loaded_quant = _normalize_quantization_mode(
                    str(reload_reply.get("quantization", quant_mode) or quant_mode), default=str(quant_mode)
                )
                print(f"[Startup] target_reload_ok quant={loaded_quant}", flush=True)
                if debug:
                    print(
                        f"[Target Sync] reload_ok model={reload_reply.get('loaded_model', expected_model)} "
                        f"changed={reload_reply.get('changed', True)} quant={loaded_quant}"
                    )
                return {
                    "selected_quantization": str(loaded_quant),
                    "attempted_quantizations": attempted_quants,
                    "fallback_applied": bool(str(loaded_quant) != str(expected_quant) or len(attempted_quants) > 1),
                    "status": "reloaded",
                }
            message = str(reload_reply.get("message", reload_reply))
            failures.append(f"{quant_mode}:{message}")
            if not _is_target_reload_memory_related_error(message):
                break
        recommendation = (
            "Try a smaller target model, run with less concurrent GPU load, "
            "or move target profiling to a larger-memory GPU."
        )
        raise RuntimeError(
            "target reload failed after quantization fallback. "
            f"attempted={attempted_quants}, failures={failures}. {recommendation}"
        )
    finally:
        sock.settimeout(prev_timeout)


def _connect_target_with_retry(
    host: str,
    port: int,
    max_attempts: int = 8,
    base_delay_s: float = 0.5,
    connect_timeout_s: float = 10.0,
):
    max_attempts = max(1, int(max_attempts))
    base_delay_s = max(0.05, float(base_delay_s))
    connect_timeout_s = max(1.0, float(connect_timeout_s))
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"[Startup] target_connecting attempt {attempt}/{max_attempts} {host}:{port}", flush=True)
            sock = socket.create_connection((host, port), timeout=connect_timeout_s)
            sock.settimeout(max(30.0, connect_timeout_s))
            print("[Startup] target_connected", flush=True)
            return sock
        except Exception as exc:
            last_exc = exc
            print(f"[Startup] target_connect_failed attempt {attempt}/{max_attempts}: {exc}", flush=True)
            if attempt >= max_attempts:
                break
            wait_s = min(5.0, base_delay_s * (2 ** (attempt - 1)))
            time.sleep(wait_s)
    raise RuntimeError(f"target connection failed after {max_attempts} attempts: {last_exc}")


def _estimate_server_only_init_timeout_seconds(
    server_draft_profile_auto: bool,
    server_draft_profile_force_refresh: bool,
    server_draft_profile_model_calls_per_count: int,
    server_draft_profile_width_list: str,
) -> float:
    """
    Estimate the timeout for waiting on the server_only_init response.
    - When auto profile and force refresh are enabled, regenerating the width profile on the target can take a long time,
      so this must be larger than the default socket timeout (30s).
    """
    if not bool(server_draft_profile_auto):
        return 30.0
    if not bool(server_draft_profile_force_refresh):
        # , miss .
        return 300.0
    # force-refresh: width_count * model_calls_per_count
    width_count = 15
    try:
        parsed = [tok.strip() for tok in str(server_draft_profile_width_list).split(",") if tok.strip()]
        width_count = max(1, len(parsed))
    except Exception:
        width_count = 15
    calls = max(1, int(server_draft_profile_model_calls_per_count))
    estimated = 120.0 + float(width_count * calls) * 0.25
    return float(min(3600.0, max(300.0, estimated)))


def _resolve_cloud_transfer_costs(
    user_communication_cost_per_gb: float = None,
    cloud_outbound_cost_per_gb: float = None,
):
    """Resolve user/cloud transfer pricing for directional charging."""
    user_comm_cost = (
        float(user_communication_cost_per_gb)
        if user_communication_cost_per_gb is not None
        else 0.09
    )
    outbound = (
        float(cloud_outbound_cost_per_gb)
        if cloud_outbound_cost_per_gb is not None
        else 0.09
    )
    return max(0.0, user_comm_cost), max(0.0, outbound)


class DraftRunner:
    def __init__(self, draft_model: torch.nn.Module, tokenizer: AutoTokenizer, debug: bool = False, profile_data: dict = None, draft_per_sec_cost: float = 0.0, target_per_sec_cost: float = 0.0, draft_electricity_cost_per_kwh: float = 0.2, user_communication_cost_per_gb: float = 0.09, cloud_outbound_cost_per_gb: float = 0.09, cost_sensitivity: float = 0.0, enable_gpu_monitor: bool = False, gpu_monitor_interval: float = 0.05, enable_cpu_monitor: bool = False, fix_gpu_clock: bool = False, gpu_graphics_clock_mhz: int = None, gpu_memory_clock_mhz: int = None, opt_tree: bool = False, no_draft_cost: bool = False, objective_metric: str = "total_cost", accept_length_margin: float = 0.05, objective_selection_mode: str = "blend", constraint_target: str = "metric", metric_constraint_per_token: float = None, min_tps_constraint: float = None, bill_draft_as_target_gpu: bool = False):
        
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        # Draft tree stable KV
        self.draft_stable_kv = None
        # Proactive drafting KV ( draft_stable_kv )
        self.proactive_kv = None
        self.debug = debug
        self.enable_gpu_monitor = enable_gpu_monitor
        self.gpu_monitor = None
        if self.enable_gpu_monitor:
            self.gpu_monitor = GPUMonitor(
                interval=gpu_monitor_interval,
                fix_gpu_clock=bool(fix_gpu_clock),
                graphics_clock=gpu_graphics_clock_mhz,
                memory_clock=gpu_memory_clock_mhz,
                debug=debug
            )
        # CPU 
        self.enable_cpu_monitor = enable_cpu_monitor
        self.cpu_power_monitor = None
        if self.enable_cpu_monitor:  # CPU CPU
            # CPU 0.5 interval (powerstat )
            cpu_monitor_interval = max(0.5, gpu_monitor_interval)
            self.cpu_power_monitor = CPUPowerMonitor(
                interval=cpu_monitor_interval,
                debug=debug
            )
            if debug:
                print(f"[draft] CPU power monitor initialized (interval: {cpu_monitor_interval}, requested: {gpu_monitor_interval})")
        else:
            if debug:
                print(f"[draft] CPU power monitor disabled (enable_cpu_monitor: {self.enable_cpu_monitor})")
        # width draft_model.model
        self.accumulated_width_times = {}  # {width: [time1, time2, ...]} - draft_model.model
        # model.model()
        self.accumulated_model_total_time = 0.0
        # model.model()
        self.accumulated_expected_model_total_time = 0.0
        # (width model_call_avg_time_ms )
        self.profile_data = profile_data  # {width: {"model_call_avg_time_ms": ...}, ...}
        # Target (width_depth_nnodes avg_time_ms )
        self.target_profile_data = None  # {"nnodes_Z": {"max_nnodes": Z, "avg_time_ms": ...}, ...}
        # tree 
        # max_nnodes (draft() )
        self.prev_tree_final_nnodes = None  # tree top_index (final_nnodes), None
        self.prev_tree_depth = 0  # tree depth
        self.prev_tree_total_target_time = 0.4  # tree (target , ), 0.4
        self.prev_tree_transfer_time = None  # tree ( , draft_to_target + target_to_draft)
        self.prev_tree_accept_length = 1.0  # tree
        # ( , )
        self.per_token_draft_to_target_transfer_time = 0.0  # Draft Target
        self.per_token_target_to_draft_transfer_time = 0.0  # Target Draft
        self.per_token_draft_to_target_bytes = 0.0  # Draft -> Target
        self.per_token_target_to_draft_bytes = 0.0  # Target -> Draft
        self.user_communication_cost_per_gb = float(user_communication_cost_per_gb or 0.0)
        self.cloud_outbound_cost_per_gb = float(cloud_outbound_cost_per_gb or 0.0)
        self.draft_per_sec_cost = draft_per_sec_cost
        self.target_per_sec_cost = target_per_sec_cost
        self.draft_electricity_cost_per_kwh = max(0.0, float(draft_electricity_cost_per_kwh or 0.0))
        # Cost sensitivity weight in [0, 1].
        self.cost_sensitivity = cost_sensitivity
        # opt-tree
        self.opt_tree = bool(opt_tree)
        # draft tree
        self.no_draft_cost = bool(no_draft_cost)
        self.bill_draft_as_target_gpu = bool(bill_draft_as_target_gpu)
        # objective metric:
        # - cost
        # - total_cost
        # - api_cost
        # - draft_energy
        # - target_energy
        self.objective_metric = _normalize_objective_metric(objective_metric)
        self.objective_selection_mode = str(objective_selection_mode).lower() if objective_selection_mode is not None else "blend"
        if self.objective_selection_mode not in {"blend", "constraint"}:
            self.objective_selection_mode = "blend"
        self.constraint_target = str(constraint_target).lower() if constraint_target is not None else "metric"
        if self.constraint_target not in {"metric", "tps"}:
            self.constraint_target = "metric"
        self.metric_constraint_per_token = (
            float(metric_constraint_per_token)
            if metric_constraint_per_token is not None
            else None
        )
        self.min_tps_constraint = (
            float(min_tps_constraint)
            if min_tps_constraint is not None and float(min_tps_constraint) > 0
            else None
        )
        # Tree objective rate
        # - total_cost : dollar/sec (GPU power EMA kWh/sec -> $/sec )
        # - api_cost : draft objective 0, target dollar/sec
        # - energy_* : draft/target=kWh/sec (GPU power_draw_w EMA)
        self.draft_objective_rate_per_sec = (
            float(draft_per_sec_cost)
            if (
                self.objective_metric == "total_cost"
                or (
                    self.objective_metric == "api_cost"
                    and bool(self.bill_draft_as_target_gpu)
                )
            )
            else 0.0
        )
        self.target_objective_rate_per_sec = (
            float(target_per_sec_cost) if self.objective_metric in {"total_cost", "api_cost"} else 0.0
        )
        # Warmup reference ( objective )
        self.reference_tps = 1.0
        self.reference_cost_per_token = 1.0
        self.reference_objective_per_token = 1.0
        # Depth ( tree )
        self.accumulated_depth_stats = {}  # {depth: {"prev_sum_expected_accepted_length": [...], "sum_expected_accepted_length": [...], "per_token_latency": [...], "per_token_cost": [...], "objective_value": [...]}}
        self.accumulated_width_algorithm_times = []  # width 
        self.accumulated_nnodes_algorithm_times = []  # final_nnodes 
        # ( 1 ) : print /
        self.question_width_times = {}  # {width: [time1, time2, ...]}
        self.question_model_total_time = 0.0
        self.question_expected_model_total_time = 0.0
        self.question_width_algorithm_times = []
        self.question_nnodes_algorithm_times = []
        # tree( step) expected accept length (tree.sum_expected_accepted_length)
        self.last_sum_expected_accepted_length = None
        # tree Tree accept_length_scale (scaled_expected )
        self.last_accept_length_scale_used = 1.0
        # tree_build (ms)
        self.last_tree_timing_breakdown = {
            "tree_model_forward_ms": 0.0,
            "tree_width_algo_ms": 0.0,
            "tree_nnodes_algo_ms": 0.0,
            "tree_mask_build_ms": 0.0,
            "tree_finalize_ms": 0.0,
            "tree_budget_wait_ms": 0.0,
        }
        # proactive tree ( step target verification + transfer )
        self.proactive_budget_ms = None
        # profile_data/target_profile_data ( / )
        # - draft_time_ratio = draft_time_ms / predicted_time_ms (width )
        # - target_verification_ratio = target_verification_time_ms / predicted_time_ms (width/depth/nnodes )
        self.draft_time_ratio_sum = 0.0
        self.draft_time_ratio_count = 0
        self.target_verification_ratio_sum = 0.0
        self.target_verification_ratio_count = 0
        # expected_accept_length
        # - scale = ( actual_accept_length) / ( expected_accept_length_raw)
        self.accept_length_actual_sum = 0.0
        self.accept_length_expected_sum = 0.0
        # reference accept_length_scale
        self.accept_length_scale_override = None
        # expected accept length ( 5%)
        self.accept_length_margin = max(0.0, min(0.99, float(accept_length_margin)))
        # target profile lookup
        self.target_profile_lookup_stats = {"direct_hit": 0, "nearest_hit": 0, "fallback": 0}
        # constraint fallback (Tree )
        self.constraint_fallback_stats = {
            "width_candidate_feasible": 0,
            "width_candidate_infeasible": 0,
            "width_selected_feasible": 0,
            "width_selected_fallback": 0,
            "nnodes_candidate_feasible": 0,
            "nnodes_candidate_infeasible": 0,
            "nnodes_selected_feasible": 0,
            "nnodes_selected_fallback": 0,
        }
        # online profile update runtime options/paths
        self.draft_profile_file = None
        self.target_profile_file = None
        self.online_profile_update_enabled = True
        self.online_profile_lr = float(ONLINE_PROFILE_LR_DEFAULT)
        self.server_only_mode = False

    def get_draft_objective_rate_per_sec(self) -> float:
        if self.no_draft_cost:
            return 0.0
        return float(self.draft_objective_rate_per_sec)

    def uses_total_cost_objective(self) -> bool:
        return self.objective_metric == "total_cost"

    def uses_api_cost_objective(self) -> bool:
        return self.objective_metric == "api_cost"

    def uses_any_cost_objective(self) -> bool:
        return self.objective_metric in {"total_cost", "api_cost"}

    def uses_draft_energy_objective(self) -> bool:
        return self.objective_metric == "draft_energy"

    def uses_server_only_target_energy_sum(self) -> bool:
        return bool(self.server_only_mode) and self.objective_metric == "target_energy"

    def uses_draft_energy_profile(self) -> bool:
        """
        Objectives that require a draft-side energy profile (gpu_power/energy).
        - total_cost: uses draft energy (kWh) * electricity cost ($/kWh) (no_draft_cost=False)
        - draft_energy: uses draft energy itself as the objective
        """
        if self.uses_total_cost_objective():
            # server-side shared-GPU billing draft target_per_sec_cost
            # draft energy profile(gpu power) .
            return (not bool(self.no_draft_cost)) and (not bool(getattr(self, "bill_draft_as_target_gpu", False)))
        return self.uses_draft_energy_objective() or self.uses_server_only_target_energy_sum()

    def uses_target_energy_objective(self) -> bool:
        return self.objective_metric == "target_energy"

    def get_target_objective_rate_per_sec(self) -> float:
        return float(self.target_objective_rate_per_sec)

    def get_sensitivity_alpha(self) -> float:
        """Return cost-sensitivity directly as alpha in [0, 1]."""
        try:
            alpha = float(self.cost_sensitivity)
        except Exception:
            alpha = 0.0
        return max(0.0, min(1.0, alpha))

    def get_reference_latency_per_token(self) -> float:
        return 1.0 / max(1e-9, float(self.reference_tps))

    def get_reference_objective_per_token(self) -> float:
        return max(1e-12, float(self.reference_objective_per_token))

    def _extract_gpu_power_avg_w(self, gpu_stats: dict) -> Optional[float]:
        if not isinstance(gpu_stats, dict) or not gpu_stats:
            return None
        try:
            gpu_entry = gpu_stats.get("gpu_0")
            if gpu_entry is None:
                gpu_entry = next(iter(gpu_stats.values()))
            power_info = gpu_entry.get("power_draw_w", {}) if isinstance(gpu_entry, dict) else {}
            power_avg_w = power_info.get("avg") if isinstance(power_info, dict) else None
            if power_avg_w is None:
                return None
            power_avg_w = float(power_avg_w)
            if not np.isfinite(power_avg_w) or power_avg_w <= 0:
                return None
            return power_avg_w
        except Exception:
            return None

    def update_draft_objective_rate_from_gpu(self, gpu_stats: dict, require_valid: bool = False):
        """
        Update the draft objective rate using the average GPU power_draw.
        - total_cost: kWh/sec -> $/sec after conversion to EMA
        - draft_energy: kWh/sec EMA
        """
        if not (
            self.uses_total_cost_objective()
            or self.uses_draft_energy_objective()
            or self.uses_server_only_target_energy_sum()
        ):
            return
        power_avg_w = self._extract_gpu_power_avg_w(gpu_stats)
        if power_avg_w is None:
            if require_valid:
                raise RuntimeError(
                    "GPU power_draw measurements are missing. total_cost requires GPU monitoring data (power_draw_w.avg)."
                )
            return
        measured_kwh_per_sec = power_avg_w / 3600000.0
        if self.uses_total_cost_objective():
            measured_rate = measured_kwh_per_sec * float(self.draft_electricity_cost_per_kwh)
        else:
            measured_rate = measured_kwh_per_sec
        if self.draft_objective_rate_per_sec <= 0:
            self.draft_objective_rate_per_sec = float(measured_rate)
        else:
            alpha = 0.2
            self.draft_objective_rate_per_sec = (
                alpha * float(measured_rate) + (1.0 - alpha) * self.draft_objective_rate_per_sec
            )

    def update_target_objective_rate(self, energy_rate_per_sec: Optional[float]):
        """When the target energy objective is turned on, the target rate (kWh/s) is updated to EMA."""
        if not self.uses_target_energy_objective():
            return
        if energy_rate_per_sec is None:
            return
        try:
            measured_kwh_per_sec = float(energy_rate_per_sec)
            if measured_kwh_per_sec <= 0:
                return
            if self.target_objective_rate_per_sec <= 0:
                self.target_objective_rate_per_sec = measured_kwh_per_sec
            else:
                alpha = 0.2
                self.target_objective_rate_per_sec = (
                    alpha * measured_kwh_per_sec + (1.0 - alpha) * self.target_objective_rate_per_sec
                )
        except Exception:
            return

    def get_draft_time_ratio_mean(self):
        """Cumulative average of draft_time_ms / predicted_time_ms (None if none)"""
        if self.draft_time_ratio_count <= 0:
            return None
        return float(self.draft_time_ratio_sum / self.draft_time_ratio_count)

    def get_target_verification_ratio_mean(self):
        """Cumulative average of target_verification_time_ms / predicted_time_ms (None if none)"""
        if self.target_verification_ratio_count <= 0:
            return None
        return float(self.target_verification_ratio_sum / self.target_verification_ratio_count)

    def get_accept_length_ratio_mean(self):
        """(accumulated actual_accept_length) / (accumulated expected_accept_length_raw)"""
        if self.accept_length_expected_sum <= 0:
            return None
        return float(self.accept_length_actual_sum / self.accept_length_expected_sum)
    
    def reset_kv(self):
        """KV cache reset"""
        self.draft_stable_kv = None

    def reset_proactive_kv(self):
        """Proactive KV cache reset"""
        self.proactive_kv = None
    
    def reset_timing_stats(self, reset_global: bool = False):
        """Stat Reset
        - reset_global=False: reset only per-question statistics (one question) (keep accumulated values for the full run)
        - reset_global=True : reset accumulated values for the full run as well (for example, after warmup)
        """
        self.question_width_times = {}
        self.question_model_total_time = 0.0
        self.question_expected_model_total_time = 0.0
        self.question_width_algorithm_times = []
        self.question_nnodes_algorithm_times = []

        if reset_global:
            self.accumulated_width_times = {}
            self.accumulated_model_total_time = 0.0
            self.accumulated_expected_model_total_time = 0.0
            self.accumulated_width_algorithm_times = []
            self.accumulated_nnodes_algorithm_times = []
            # warmup
            self.draft_time_ratio_sum = 0.0
            self.draft_time_ratio_count = 0
            self.target_verification_ratio_sum = 0.0
            self.target_verification_ratio_count = 0
            self.accept_length_actual_sum = 0.0
            self.accept_length_expected_sum = 0.0
            self.target_profile_lookup_stats = {"direct_hit": 0, "nearest_hit": 0, "fallback": 0}
            self.constraint_fallback_stats = {
                "width_candidate_feasible": 0,
                "width_candidate_infeasible": 0,
                "width_selected_feasible": 0,
                "width_selected_fallback": 0,
                "nnodes_candidate_feasible": 0,
                "nnodes_candidate_infeasible": 0,
                "nnodes_selected_feasible": 0,
                "nnodes_selected_fallback": 0,
            }
    
    def print_timing_stats(self):
        """Output average execution time of draft_model.model call by accumulated width"""
        if not self.question_width_times:
            return
        
        print("\n" + "="*80)
        print("draft_model.model call timing statistics by width (accumulated for the question)")
        print("="*80)
        print(f"{'Width':<10} {'Call Count':<12} {'Total Time (ms)':<15} {'Average Time (ms)':<15} {'Min Time (ms)':<15} {'Max Time (ms)':<15}")
        print("-"*80)
        
        for width in sorted(self.question_width_times.keys()):
            times = self.question_width_times[width]
            count = len(times)
            total_time_ms = sum(times) * 1000
            avg_time_ms = (sum(times) / count) * 1000 if count > 0 else 0
            min_time_ms = min(times) * 1000
            max_time_ms = max(times) * 1000
            
            print(f"{width:<10} {count:<12} {total_time_ms:<15.3f} {avg_time_ms:<15.3f} {min_time_ms:<15.3f} {max_time_ms:<15.3f}")
        
        print("="*80)
        
        print(f"\nModel call time totals:")
        print(f"  Measured time: {self.question_model_total_time * 1000:.3f} ms")
        print(f"  Expected time (profiled): {self.question_expected_model_total_time * 1000:.3f} ms")
        if self.question_model_total_time > 0:
            diff_ms = (self.question_expected_model_total_time - self.question_model_total_time) * 1000
            diff_percent = (diff_ms / (self.question_model_total_time * 1000)) * 100
            print(f"  Difference: {diff_ms:+.3f} ms ({diff_percent:+.2f}%)")
        print("="*80 + "\n")

    def process_tree_mask(self, tree_attention_mask: torch.Tensor, init_len: int) -> torch.Tensor:
        # tree_attention_mask (current_width, past_cols + current_width)
        # past_cols depth
        # init_len
        # (current_width, init_len + past_cols + current_width)
        
        current_width = tree_attention_mask.size(0)
        tree_mask_cols = tree_attention_mask.size(1)
        past_cols = tree_mask_cols - current_width
        
        # mask (current_width, init_len)
        attention_mask = torch.full((current_width, init_len), 0, device=tree_attention_mask.device)
        
        # tree_attention_mask : 0 -inf , 1 0
        tree_mask = torch.where(tree_attention_mask == 0, torch.finfo(torch.float32).min, 0.0)
        
        # tree_attention_mask past_cols , init_len
        # tree_mask
        # concat: (current_width, init_len) + (current_width, past_cols + current_width)
        attention_mask = torch.cat([attention_mask, tree_mask], dim=-1)
        
        # (1, 1, current_width, init_len + past_cols + current_width)
        attention_mask = attention_mask[None, None, :, :]
        return attention_mask

    @torch.no_grad()
    def draft(self, input_ids: torch.Tensor, nodes: int, max_depth: int, print_time: bool = False, print_tree: bool = False, tokenizer=None, per_token_probability_bound: float = 0.0, per_path_probability_bound: float = 0.0, min_width: int = 1, fixed_width: bool = False, fixed_nnodes: bool = False, fixed_depth: bool = False, use_proactive_kv: bool = False, track_stats: bool = True, stop_flag: threading.Event = None, fixed_width_value: int = None, proactive_time_budget_sec: float = None, proactive_continue_event: threading.Event = None, proactive_use_probability: float = None, proactive_depth_stats: dict = None, proactive_disable_budget: bool = False):
        # breakpoint()
        device = self.draft_model.lm_head.weight.device
        len_posi = input_ids.shape[1] - 1
        kv_attr = "proactive_kv" if use_proactive_kv else "draft_stable_kv"
        kv_state = getattr(self, kv_attr)
        
        # Stable KV : KV
        if kv_state is not None:
            kv_len = kv_state[0][0].shape[2]  # KV
            new_tokens = input_ids[:, kv_len:].to(device)  # accepted tokens + next_token
            seq_len = new_tokens.shape[1]
            if seq_len == 0:
                # : _next_token == _next_token <= ? .
                # draft_stable_kv ( _next_token) hidden state
                if self.debug:
                    print(f"[DRAFT-INFO] Empty new_tokens (prev_next_token == new_next_token). Recomputing last hidden state.")
                # draft_stable_kv ( _next_token) past_key_values
                trimmed_kv = tuple(
                    (kv[0][:, :, :-1, :], kv[1][:, :, :-1, :]) 
                    for kv in kv_state
                )
                # hidden state forward
                # : _next_token == _next_token
                # input_ids[-1] = _next_token = _next_token
                last_token_id = input_ids[0, -1:].unsqueeze(0).to(device)
                draft_outputs = self.draft_model.model(
                    input_ids=last_token_id,
                    past_key_values=trimmed_kv,
                    position_ids=torch.tensor([[kv_len - 1]], device=device, dtype=torch.long),
                    return_kv=True,
                    is_draft=True,
                )
                # draft_stable_kv ( _next_token )
                past_key_values = kv_state
            else:  # .
                pos_ids = torch.arange(kv_len, kv_len + seq_len, device=device, dtype=torch.long)
                draft_outputs = self.draft_model.model(
                    input_ids=new_tokens,
                    past_key_values=kv_state,
                    position_ids=pos_ids,
                    return_kv=True,
                    is_draft=True,
                )
        else:
            full = input_ids.to(device)
            seq_len = full.shape[1]
            if seq_len == 0:
                raise RuntimeError("DraftRunner.draft received empty input_ids on initial call")
            pos_ids = torch.arange(0, seq_len, device=device, dtype=torch.long)
            draft_outputs = self.draft_model.model(
                input_ids=full,
                position_ids=pos_ids,
                return_kv=True,
                is_draft=True,
            )
        
        # Base KV (tree )
        setattr(self, kv_attr, draft_outputs[1])
        past_key_values = getattr(self, kv_attr)
        init_len = past_key_values[0][0].size(2)
        
        last_hidden = draft_outputs[0][:, -1]
        last_headout = self.draft_model.lm_head(last_hidden)

        # prev_tree_final_width None max_nnodes
        prev_tree_final_width = self.prev_tree_final_nnodes if self.prev_tree_final_nnodes is not None else nodes
        
        if self.accept_length_scale_override is not None:
            accept_length_scale = float(self.accept_length_scale_override)
        else:
            accept_length_scale = self.get_accept_length_ratio_mean() if self.get_accept_length_ratio_mean() is not None else 1.0

        tree_objective_metric = (
            "_combined_energy"
            if self.uses_server_only_target_energy_sum()
            else self.objective_metric
        )
        tree = Tree(
            nodes, 
            last_hidden.device, 
            max_depth, 
            per_token_probability_bound=per_token_probability_bound, 
            per_path_probability_bound=per_path_probability_bound, 
            fixed_width=fixed_width_value if fixed_width_value is not None else (nodes if fixed_width else None),
            fixed_nnodes=fixed_nnodes,
            fixed_depth=fixed_depth,
            profile_data=self.profile_data,
            target_profile_data=self.target_profile_data,
            target_time_scale=self.get_target_verification_ratio_mean() if self.get_target_verification_ratio_mean() is not None else 1.0,
            accept_length_scale=accept_length_scale,
            accept_length_margin=self.accept_length_margin,
            objective_selection_mode=self.objective_selection_mode,
            constraint_target=self.constraint_target,
            metric_constraint_per_token=self.metric_constraint_per_token,
            min_tps_constraint=self.min_tps_constraint,
            draft_per_sec_cost=self.get_draft_objective_rate_per_sec(),
            target_per_sec_cost=self.get_target_objective_rate_per_sec(),
            cost_sensitivity=self.cost_sensitivity,
            reference_tps=self.reference_tps,
            reference_objective_per_token=self.reference_objective_per_token,
            objective_metric=tree_objective_metric,
            no_draft_cost=self.no_draft_cost,
            opt_tree=self.opt_tree,
            min_width=min_width,
            per_token_draft_to_target_transfer_time=self.per_token_draft_to_target_transfer_time,
            per_token_target_to_draft_transfer_time=self.per_token_target_to_draft_transfer_time,
            per_token_draft_to_target_bytes=self.per_token_draft_to_target_bytes,
            per_token_target_to_draft_bytes=self.per_token_target_to_draft_bytes,
            user_communication_cost_per_gb=self.user_communication_cost_per_gb,
            cloud_outbound_cost_per_gb=self.cloud_outbound_cost_per_gb,
            stop_flag=stop_flag,
            proactive_time_budget_sec=proactive_time_budget_sec,
            proactive_continue_event=proactive_continue_event,
            proactive_use_probability=proactive_use_probability,
            proactive_depth_stats=proactive_depth_stats,
            proactive_disable_budget=proactive_disable_budget,
        )

        logits = last_headout.unsqueeze(0)
        end = False
        draft_time = None  # draft model ( update None)
        mask_build_time_sec = 0.0
        while not end:  # tree draft tree .
            if stop_flag is not None and stop_flag.is_set():
                return None
            draft_time_scale = self.get_draft_time_ratio_mean() if self.get_draft_time_ratio_mean() is not None else 1.0
            tree_output = tree.update(
                torch.softmax(logits.to(last_hidden.device), dim=-1, dtype=torch.float32),
                print_tree=print_tree,
                tokenizer=tokenizer,
                draft_time=draft_time,
                draft_time_scale=draft_time_scale,
            )

            input_ids_step = tree_output["input_ids"].unsqueeze(0)
            position_ids = tree_output["position_ids"] + len_posi   # [DH] index -> index ( input_ids )
            if tree_output["is_final"]:
                break
            mask_build_start = time.time()
            tree_attention_mask_with_kv = self.process_tree_mask(tree_output["attention_mask"], init_len)
            mask_build_time_sec += max(0.0, float(time.time() - mask_build_start))
            
            # draft_model.model
            if isinstance(input_ids_step, torch.Tensor) and input_ids_step.is_cuda:
                torch.cuda.synchronize()
            model_start_time = time.time()
            draft_outputs = self.draft_model.model(
                input_ids=input_ids_step,
                position_ids=position_ids,
                past_key_values=past_key_values,
                tree_attention_mask=tree_attention_mask_with_kv,
                return_kv=True,
                is_draft=True,
            )
            # breakpoint()
            if isinstance(draft_outputs[0], torch.Tensor) and draft_outputs[0].is_cuda:
                torch.cuda.synchronize()
            draft_time = time.time() - model_start_time

            # draft_time / profile_data 
            # width forward (=input_ids_step )
            try:
                width_call = int(input_ids_step.shape[1])
            except Exception:
                width_call = None
            if track_stats and width_call is not None and self.profile_data is not None:
                width_str = str(width_call)
                predicted_ms = self.profile_data.get(width_str, {}).get("model_call_avg_time_ms", None)
                if predicted_ms is not None and predicted_ms > 0:
                    ratio = (float(draft_time) * 1000.0) / float(predicted_ms)
                    if ratio == ratio and ratio > 0:  # NaN
                        self.draft_time_ratio_sum += float(ratio)
                        self.draft_time_ratio_count += 1
            # Tree KV (local variable)
            past_key_values = draft_outputs[1]
            
            last_hidden = draft_outputs[0]
            last_headout = self.draft_model.lm_head(last_hidden)
            logits = last_headout

        if track_stats:
            # Tree (draft_model.model )
            for width, times in tree.width_times.items():
                if width not in self.question_width_times:
                    self.question_width_times[width] = []
                self.question_width_times[width].extend(times)
                # (run)
                if width not in self.accumulated_width_times:
                    self.accumulated_width_times[width] = []
                self.accumulated_width_times[width].extend(times)
            
            self.question_model_total_time += tree.draft_total_time
            self.accumulated_model_total_time += tree.draft_total_time
            # model.model()
            self.question_expected_model_total_time += tree.expected_draft_total_time
            self.accumulated_expected_model_total_time += tree.expected_draft_total_time
            
            # Depth (tree.depth_stats )
            for depth, stats in tree.depth_stats.items():
                if depth not in self.accumulated_depth_stats:
                    self.accumulated_depth_stats[depth] = {
                        "prev_sum_expected_accepted_length": [],
                        "sum_expected_accepted_length": [],
                        "per_token_latency": [],
                        "per_token_cost": [],
                        "objective_value": []
                    }
                for key in ["prev_sum_expected_accepted_length", "sum_expected_accepted_length", "per_token_latency", "per_token_cost", "objective_value"]:
                    self.accumulated_depth_stats[depth][key].extend(stats[key])
            
            if hasattr(tree, 'width_algorithm_times') and tree.width_algorithm_times:
                self.question_width_algorithm_times.extend(tree.width_algorithm_times)
                self.accumulated_width_algorithm_times.extend(tree.width_algorithm_times)
            if hasattr(tree, 'nnodes_algorithm_times') and tree.nnodes_algorithm_times:
                self.question_nnodes_algorithm_times.extend(tree.nnodes_algorithm_times)
                self.accumulated_nnodes_algorithm_times.extend(tree.nnodes_algorithm_times)
            if hasattr(tree, "target_profile_lookup_stats"):
                for k, v in tree.target_profile_lookup_stats.items():
                    self.target_profile_lookup_stats[k] = self.target_profile_lookup_stats.get(k, 0) + int(v)
            if hasattr(tree, "constraint_decision_stats"):
                for k, v in tree.constraint_decision_stats.items():
                    self.constraint_fallback_stats[k] = self.constraint_fallback_stats.get(k, 0) + int(v)

        self.last_tree_timing_breakdown = {
            "tree_model_forward_ms": float(max(0.0, tree.draft_total_time) * 1000.0),
            "tree_width_algo_ms": float(sum(tree.width_algorithm_times) * 1000.0) if hasattr(tree, "width_algorithm_times") else 0.0,
            "tree_nnodes_algo_ms": float(sum(tree.nnodes_algorithm_times) * 1000.0) if hasattr(tree, "nnodes_algorithm_times") else 0.0,
            "tree_mask_build_ms": float(max(0.0, mask_build_time_sec) * 1000.0),
            "tree_finalize_ms": float(max(0.0, getattr(tree, "last_finalize_time_sec", 0.0)) * 1000.0),
            "tree_budget_wait_ms": float(max(0.0, getattr(tree, "proactive_budget_wait_sec", 0.0)) * 1000.0),
            "proactive_expand_continue_count": int(getattr(tree, "proactive_expand_continue_count", 0)),
            "proactive_expand_pause_count": int(getattr(tree, "proactive_expand_pause_count", 0)),
            "proactive_finalize_early_count": int(getattr(tree, "proactive_finalize_early_count", 0)),
            "proactive_expected_gain_ms": float(max(0.0, getattr(tree, "proactive_expected_gain_sec", 0.0)) * 1000.0),
            "proactive_expected_loss_ms": float(max(0.0, getattr(tree, "proactive_expected_loss_sec", 0.0)) * 1000.0),
            "proactive_last_expand_decision": getattr(tree, "proactive_last_expand_decision", None),
            "proactive_expand_depth_counts": dict(getattr(tree, "proactive_expand_depth_counts", {}) or {}),
            "proactive_expected_gain_by_depth_ms": {
                str(k): float(v) * 1000.0 for k, v in (getattr(tree, "proactive_expected_gain_by_depth", {}) or {}).items()
            },
            "proactive_expected_loss_by_depth_ms": {
                str(k): float(v) * 1000.0 for k, v in (getattr(tree, "proactive_expected_loss_by_depth", {}) or {}).items()
            },
        }

       
        # Tree KV 
        # final_nnodes model.model() final_nnodes
        final_nnodes = tree.final_nnodes
        # tree expected accept length (caller / )
        try:
            expected_accept_length = float(tree.sum_expected_accepted_length)
        except Exception:
            expected_accept_length = None
        # depth width
        depth_widths = list(tree.depth_widths)

        # proactive meta: node depth/col/path_prob (final )
        node_meta = None
        if isinstance(tree_output, dict) and tree_output.get("is_final"):
            rows = tree_output.get("rows", None)
            cols = tree_output.get("cols", None)
            if rows is not None and cols is not None:
                try:
                    path_probs = tree.weight_matrix[rows, cols].detach().to("cpu").tolist()
                except Exception:
                    path_probs = None
                node_meta = {
                    "rows": rows.detach().to("cpu").tolist() if isinstance(rows, torch.Tensor) else rows,
                    "cols": cols.detach().to("cpu").tolist() if isinstance(cols, torch.Tensor) else cols,
                    "path_probs": path_probs,
                }

        return (
            input_ids_step,
            position_ids,
            tree_output["attention_mask"],
            tree_output["parent_last"],
            tree.depth,
            final_nnodes,
            depth_widths,
            node_meta,
            expected_accept_length,
            float(accept_length_scale),
        )


@torch.inference_mode()
def build_tree_with_next_token(
    runner: DraftRunner,
    input_ids: torch.Tensor,
    nodes: int,
    max_depth: int,
    next_token_id: int,
    tokenizer: AutoTokenizer,
    debug: bool = False,
    print_tree: bool = False,
    per_token_probability_bound: float = 0.0,
    per_path_probability_bound: float = 0.0,
    min_width: int = 1,
    fixed_width: bool = False,
    fixed_width_value: int = None,
    fixed_nnodes: bool = False,
    fixed_depth: bool = False,
):
    # tree
    if runner.draft_stable_kv is not None:
        kv_len = runner.draft_stable_kv[0][0].shape[2]  # draft_stable_kv KV
        input_ids_len = input_ids.shape[1]
        # : draft_stable_kv = input_ids + next_token
        # input_ids = input_ids ( accepted_tokens )
        if kv_len > input_ids_len + 1:
            # draft_stable_kv input_ids + next_token , .
            if debug:
                print(f"[DRAFT-DEBUG] KV sync check failed: kv_len={kv_len} > input_ids_len+1={input_ids_len+1}, resetting")
            runner.draft_stable_kv = None
    
    next_token = torch.tensor([[next_token_id]], device=input_ids.device, dtype=torch.long)
    cat_input = torch.cat((input_ids, next_token), dim=1)
    if runner.draft_stable_kv is not None:
        kv_len = runner.draft_stable_kv[0][0].shape[2]
        if debug:
            print(f"[DRAFT-DEBUG] KV sync check: kv_len={kv_len}, input_ids_len={input_ids.shape[1]}, cat_input_len={cat_input.shape[1]}")
        if kv_len > cat_input.shape[1]:
            if debug:
                print(f"[DRAFT-WARN] kv_len({kv_len}) > cat_input_len({cat_input.shape[1]}). This will cause empty new_tokens. Check update_kv_with_accepted/base_input_len.")
    
    # draft , input_ids + next_token token tree . ( verify Line 226: model.draft(input_ids + next_token))
    (
        draft_input_ids,
        draft_position_ids,
        tree_attention_mask,
        parent,
        tree_depth,
        final_nnodes,
        depth_widths,
        node_meta,
        expected_accept_length,
        accept_length_scale_used,
    ) = runner.draft(
        cat_input,
        nodes,
        max_depth,
        print_tree=print_tree,
        tokenizer=tokenizer,
        per_token_probability_bound=per_token_probability_bound,
        per_path_probability_bound=per_path_probability_bound,
        min_width=min_width,
        fixed_width=fixed_width,
        fixed_width_value=fixed_width_value,
        fixed_nnodes=fixed_nnodes,
        fixed_depth=fixed_depth,
    )
    # tree expected/scale (target accept_length )
    runner.last_sum_expected_accepted_length = expected_accept_length
    runner.last_accept_length_scale_used = float(accept_length_scale_used) if accept_length_scale_used is not None else 1.0
    
    # draft next_token ( verify Line 249: next_draft = torch.cat([next_token, next_draft], dim=-1))
    postprocess_start = time.time()
    next_token_tensor = torch.tensor([[next_token_id]], device=input_ids.device, dtype=torch.long)
    draft_input_ids = torch.cat([next_token_tensor, draft_input_ids], dim=-1)
    head_pos = torch.tensor([cat_input.shape[1]-1], device=draft_position_ids.device)
    draft_position_ids = torch.cat([head_pos, draft_position_ids], dim=-1)
    tree_attention_mask = torch.cat(
        [torch.zeros(1, tree_attention_mask.size(1), dtype=tree_attention_mask.dtype, device=tree_attention_mask.device), tree_attention_mask],
        dim=0,
    )
    tree_attention_mask = torch.cat(
        [torch.ones(tree_attention_mask.size(0), 1, dtype=tree_attention_mask.dtype, device=tree_attention_mask.device), tree_attention_mask],
        dim=1,
    )

    # draft_input_ids (head + current_width) ,
    # tree_attention_mask (N, N) target .
    # tree_attention_mask (N, 1 + past_cols + (N-1)) past_cols .
    n = tree_attention_mask.size(0)
    if tree_attention_mask.size(1) != n and n > 1:
        # column(head) + (n-1) columns
        tree_attention_mask = torch.cat([tree_attention_mask[:, :1], tree_attention_mask[:, -(n - 1):]], dim=1)
    postprocess_ms = float(max(0.0, time.time() - postprocess_start) * 1000.0)
    if isinstance(getattr(runner, "last_tree_timing_breakdown", None), dict):
        runner.last_tree_timing_breakdown["tree_finalize_ms"] = (
            float(runner.last_tree_timing_breakdown.get("tree_finalize_ms", 0.0)) + postprocess_ms
        )

    return draft_input_ids, draft_position_ids, tree_attention_mask, parent, tree_depth, final_nnodes, depth_widths, node_meta


def _select_proactive_path(node_tokens, parent, node_meta):
    """Deepest node (maximum path_prob in case of tie) Returns path token sequence and path_prob."""
    if not node_meta or not node_tokens or parent is None:
        return None, None
    rows = node_meta.get("rows")
    path_probs = node_meta.get("path_probs")
    if rows is None or path_probs is None:
        return None, None
    if len(rows) != len(node_tokens) or len(parent) != len(node_tokens):
        return None, None
    max_depth = max(rows) if rows else None
    if max_depth is None:
        return None, None
    candidate_indices = [i for i, r in enumerate(rows) if r == max_depth]
    if not candidate_indices:
        return None, None
    best_idx = max(candidate_indices, key=lambda i: path_probs[i] if path_probs is not None else 0.0)
    path_indices = []
    seen = set()
    idx = best_idx
    while True:
        if idx in seen:
            break
        seen.add(idx)
        path_indices.append(idx)
        p = parent[idx]
        if p == idx or p < 0 or p >= len(parent):
            break
        idx = p
    path_indices.reverse()
    best_prob = path_probs[best_idx] if path_probs is not None and best_idx is not None else None
    return [node_tokens[i] for i in path_indices], best_prob


@torch.no_grad()
def build_proactive_tree_from_path(
    runner: DraftRunner,
    base_input_ids: torch.Tensor,
    path_tokens: List[int],
    nodes: int,
    max_depth: int,
    tokenizer: AutoTokenizer,
    debug: bool = False,
    print_tree: bool = False,
    per_token_probability_bound: float = 0.0,
    per_path_probability_bound: float = 0.0,
    min_width: int = 1,
    fixed_width: bool = False,
    fixed_width_value: int = None,
    fixed_nnodes: bool = False,
    fixed_depth: bool = False,
    stop_flag: threading.Event = None,
    head_token_holder: dict = None,
    proactive_time_budget_sec: float = None,
    proactive_continue_event: threading.Event = None,
    proactive_use_probability: float = None,
    proactive_depth_stats: dict = None,
    proactive_disable_budget: bool = False,
):
    """Creates a proactive draft tree and returns head token and tree information."""
    if not path_tokens:
        return None
    if stop_flag is not None and stop_flag.is_set():
        return None
    device = runner.draft_model.lm_head.weight.device
    # base_input_ids + path_tokens
    path_tensor = torch.tensor([path_tokens], device=base_input_ids.device, dtype=torch.long)
    proactive_input = torch.cat([base_input_ids, path_tensor], dim=-1).to(device)

    # proactive KV head
    runner.reset_proactive_kv()
    outputs = None
    # 1) draft_stable_kv (base_input_ids + current_next_token )
    base_kv = runner.draft_stable_kv
    base_len = base_input_ids.shape[1]
    total_len = proactive_input.shape[1]
    if base_kv is not None:
        kv_len = base_kv[0][0].shape[2]
        # : base_kv base_input_ids + 1(current_next_token)
        if kv_len >= base_len + 1 and kv_len <= total_len:
            new_tokens = proactive_input[:, kv_len:].to(device)
            if new_tokens.shape[1] == 0:
                # : logits
                last_token_id = proactive_input[:, -1:].to(device)
                trimmed_kv = tuple(
                    (kv[0][:, :, :-1, :], kv[1][:, :, :-1, :])
                    for kv in base_kv
                )
                pos_id = torch.tensor([[kv_len - 1]], device=device, dtype=torch.long)
                outputs = runner.draft_model.model(
                    input_ids=last_token_id,
                    past_key_values=trimmed_kv,
                    position_ids=pos_id,
                    return_kv=True,
                    is_draft=True,
                )
            else:
                pos_ids = torch.arange(kv_len, kv_len + new_tokens.shape[1], device=device, dtype=torch.long)
                outputs = runner.draft_model.model(
                    input_ids=new_tokens,
                    past_key_values=base_kv,
                    position_ids=pos_ids,
                    return_kv=True,
                    is_draft=True,
                )
    # 2) fallback: proactive_input
    if outputs is None:
        seq_len = proactive_input.shape[1]
        pos_ids = torch.arange(0, seq_len, device=device, dtype=torch.long)
        outputs = runner.draft_model.model(
            input_ids=proactive_input,
            position_ids=pos_ids,
            return_kv=True,
            is_draft=True,
        )
    runner.proactive_kv = outputs[1]
    last_hidden = outputs[0][:, -1]
    logits = runner.draft_model.lm_head(last_hidden)
    proactive_head = int(torch.argmax(logits, dim=-1).item())
    if head_token_holder is not None:
        head_token_holder["head_token"] = proactive_head
        head_token_holder["head_ready"] = True
        head_token_holder["head_ready_ts"] = time.time()

    if stop_flag is not None and stop_flag.is_set():
        return None

    # head proactive tree
    cat_input = torch.cat(
        [proactive_input, torch.tensor([[proactive_head]], device=proactive_input.device, dtype=torch.long)], dim=1
    )
    draft_result = runner.draft(
        cat_input,
        nodes,
        max_depth,
        print_tree=print_tree,
        tokenizer=tokenizer,
        per_token_probability_bound=per_token_probability_bound,
        per_path_probability_bound=per_path_probability_bound,
        min_width=min_width,
        fixed_width=fixed_width,
        fixed_width_value=fixed_width_value,
        fixed_nnodes=fixed_nnodes,
        fixed_depth=fixed_depth,
        use_proactive_kv=True,
        track_stats=False,
        stop_flag=stop_flag,
        proactive_time_budget_sec=proactive_time_budget_sec,
        proactive_continue_event=proactive_continue_event,
        proactive_use_probability=proactive_use_probability,
        proactive_depth_stats=proactive_depth_stats,
        proactive_disable_budget=proactive_disable_budget,
    )
    if draft_result is None:
        return None
    (
        draft_ids,
        draft_pos,
        tree_mask,
        parent,
        tree_depth,
        final_nnodes,
        depth_widths,
        node_meta,
        expected_accept_length,
        accept_length_scale_used,
    ) = draft_result

    # draft head (build_tree_with_next_token )
    head_tensor = torch.tensor([[proactive_head]], device=draft_ids.device, dtype=torch.long)
    draft_ids = torch.cat([head_tensor, draft_ids], dim=-1)
    head_pos = torch.tensor([cat_input.shape[1]-1], device=draft_pos.device)
    draft_pos = torch.cat([head_pos, draft_pos], dim=-1)
    tree_mask = torch.cat(
        [torch.zeros(1, tree_mask.size(1), dtype=tree_mask.dtype, device=tree_mask.device), tree_mask],
        dim=0,
    )
    tree_mask = torch.cat(
        [torch.ones(tree_mask.size(0), 1, dtype=tree_mask.dtype, device=tree_mask.device), tree_mask],
        dim=1,
    )
    n = tree_mask.size(0)
    if tree_mask.size(1) != n and n > 1:
        tree_mask = torch.cat([tree_mask[:, :1], tree_mask[:, -(n - 1):]], dim=1)

    return {
        "draft_ids": draft_ids,
        "draft_pos": draft_pos,
        "tree_mask": tree_mask,
        "parent": parent,
        "tree_depth": tree_depth,
        "final_nnodes": final_nnodes,
        "depth_widths": depth_widths,
        "node_meta": node_meta,
        "head_token": proactive_head,
        "expected_accept_length": expected_accept_length,
        "accept_length_scale_used": float(accept_length_scale_used) if accept_length_scale_used is not None else 1.0,
        "timing_breakdown": dict(getattr(runner, "last_tree_timing_breakdown", {}) or {}),
    }


def profile_width_timing(
    runner: DraftRunner,
    tokenizer: AutoTokenizer,
    max_depth: int,
    draft_model_path: str,
    device_name: str,
    draft_quantization: str,
    question_file: str,
    bench_name: str,
    width_list: List[int],
    target_model_calls_per_width: int = 1000,
    fixed_depth: bool = False,
    force_refresh: bool = False,
    profile_warmup_runs: int = PROFILE_WARMUP_RUNS,
    profile_burnin_runs: int = PROFILE_BURNIN_RUNS,
) -> dict:
    """
    Profile the average processing time by width and save it as a JSON file
    Measure by running only the draft model without communicating with the target
    
    Args:
        runner: DraftRunner instance
        tokenizer: Tokenizer
        max_depth: Maximum depth
        draft_model_path: Draft model path
        device_name: Device name (e.g. rtx5080)
        question_file: Question file path
        bench_name: Benchmark name
        width_list: Width list to profile (e.g. [50, 60, 70, 80, 90, 100])
        target_model_calls_per_width: Target number of model.model() calls to measure per width
    
    Returns:
        profile_data: Profiling data dictionary (on success), None (when skipped)
    """
    # (draft_model_path )
    # : "meta-llama/Llama-2-7b-chat-hf" -> "Llama-2-7b-chat-hf"
    model_name = draft_model_path.split("/")[-1] if "/" in draft_model_path else draft_model_path
    
    script_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(script_dir)
    profile_dir = os.path.join(_resolve_data_root(parent_dir), "profile")
    os.makedirs(profile_dir, exist_ok=True)
    profile_file = os.path.join(
        profile_dir,
        f"profile_draft_{_sanitize_key_component(device_name)}_{_sanitize_key_component(model_name)}_dq-{_sanitize_key_component(str(draft_quantization or '8bit').lower())}.json",
    )
    
    # (force_refresh=False )
    if os.path.exists(profile_file) and (not force_refresh):
        print(f"Profiling file already exists: {profile_file}")
        print("Skipping profiling step.")
        return None
    if os.path.exists(profile_file) and force_refresh:
        print(f"Refreshing existing profiling file: {profile_file}")
    
    print(f"\n{'='*80}")
    print("Starting average model.model() call-time profiling by width")
    print(f"Widths to profile: {width_list}")
    print(f"Target model call count per width: {target_model_calls_per_width}")
    print(f"{'='*80}")
    # Profiling follows the current global random/deterministic state.
    # CLI options (--seed, --deterministic) configure that state.
    print(
        "[DraftProfile] using current global random/deterministic state, "
        f"warmup_runs={max(0, int(profile_warmup_runs))}, burnin_runs={max(0, int(profile_burnin_runs))}"
    )
    original_width_list = [int(w) for w in width_list]
    width_list = sorted({int(w) for w in original_width_list if int(w) >= 2})
    if not width_list:
        raise ValueError(
            "Draft profiling requires at least one width >= 2. "
            f"requested_widths={original_width_list}"
        )
    if width_list != sorted(set(original_width_list)):
        print(
            "[DraftProfile][WARN] ignoring unsafe width values < 2: "
            f"requested={original_width_list}, effective={width_list}"
        )
    
    from fastchat.llm_judge.common import load_questions
    from fastchat.model import get_conversation_template
    
    questions = _load_benchmark_questions(bench_name, question_file)
    if len(questions) == 0:
        print("No questions available for profiling.")
        return None
    
    profile_q = questions[0]
    
    conv = _build_conversation_template_for_model(draft_model_path)
    
    # bench_name
    profile_turns = _extract_question_turns(profile_q, bench_name)
    conv.append_message(conv.roles[0], profile_turns[0])
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt() + " "
    input_ids = tokenizer([prompt]).input_ids
    input_ids_t = torch.as_tensor(input_ids).to("cuda")
    
    device = runner.draft_model.lm_head.weight.device
    require_gpu_energy_profile = bool(runner.uses_draft_energy_profile())
    if require_gpu_energy_profile and runner.gpu_monitor is None:
        raise RuntimeError(
            f"{runner.objective_metric} draft profiling requires GPU monitor, "
            "but runner.gpu_monitor is not initialized."
        )

    def _extract_gpu_stat_avg(stats: dict, field: str) -> Optional[float]:
        if not isinstance(stats, dict):
            return None
        gpu_entry = stats.get("gpu_0")
        if gpu_entry is None:
            try:
                gpu_entry = next(iter(stats.values()))
            except Exception:
                gpu_entry = None
        if not isinstance(gpu_entry, dict):
            return None
        val = gpu_entry.get(field, None)
        if isinstance(val, dict):
            val = val.get("avg", None)
        if val is None:
            return None
        try:
            val_f = float(val)
        except Exception:
            return None
        if not np.isfinite(val_f):
            return None
        return val_f
    
    # input_ids hidden state 
    runner.reset_kv()
    input_ids_for_profile = input_ids_t.to(device)
    draft_outputs = runner.draft_model.model(
        input_ids=input_ids_for_profile,
        position_ids=torch.arange(0, input_ids_for_profile.shape[1], device=device, dtype=torch.long),
        return_kv=True,
        is_draft=True,
    )
    last_hidden = draft_outputs[0][:, -1]
    last_headout = runner.draft_model.lm_head(last_hidden)
    vocab_size = last_headout.shape[-1]
    
    profile_data = {}
    
    # warmup/burnin width 1 .
    global_warmup_runs = max(0, int(profile_warmup_runs))
    global_burnin_runs = max(0, int(profile_burnin_runs))

    # width
    with torch.no_grad():
        for width_idx, width in enumerate(width_list):
            print(f"\nprofiling width={width}")
            model_call_times = []   
            gpu_power_w_samples = []  # width GPU power (W)
            gpu_util_samples = []     # width GPU utilization (%)
            gpu_graphics_clock_samples = []  # width graphics clock (MHz)
            gpu_memory_clock_samples = []    # width memory clock (MHz)
            monitor_sample_total = 0
            monitor_error_count = 0
            missing_power_count = 0
            missing_util_count = 0
            missing_gclk_count = 0
            missing_mclk_count = 0
            
            warmup_runs = global_warmup_runs if width_idx == 0 else 0
            burnin_runs = global_burnin_runs if width_idx == 0 else 0
            target_model_calls = max(1, int(target_model_calls_per_width))
            max_total_runs = warmup_runs + burnin_runs + (target_model_calls * max(1, int(max_depth)) * 5)
            run_idx = 0
            # width model call .
            while True:
                is_warmup = run_idx < warmup_runs
                is_burnin = (run_idx >= warmup_runs) and (run_idx < (warmup_runs + burnin_runs))
                # run KV logits KV
                runner.reset_kv()
                fresh_outputs = runner.draft_model.model(
                    input_ids=input_ids_for_profile,
                    position_ids=torch.arange(0, input_ids_for_profile.shape[1], device=device, dtype=torch.long),
                    return_kv=True,
                    is_draft=True,
                )
                fresh_last_hidden = fresh_outputs[0][:, -1]
                fresh_last_headout = runner.draft_model.lm_head(fresh_last_hidden)
                
                # Tree (fixed_width width )
                tree = Tree(
                    width,
                    device,
                    max_depth,
                    per_token_probability_bound=0.0,
                    per_path_probability_bound=0.0,
                    fixed_width=width,
                    fixed_nnodes=False,
                    fixed_depth=fixed_depth,
                    accept_length_margin=getattr(runner, "accept_length_margin", 0.05),
                    objective_selection_mode=getattr(runner, "objective_selection_mode", "blend"),
                    constraint_target=getattr(runner, "constraint_target", "metric"),
                    metric_constraint_per_token=getattr(runner, "metric_constraint_per_token", None),
                    min_tps_constraint=getattr(runner, "min_tps_constraint", None),
                )
                
                # initialize 
                # initialize() logits[0][-1] (batch_size, seq_len, vocab_size) .
                # fresh_last_headout (1, vocab_size) , (1, 1, vocab_size) .
                headout_1d = fresh_last_headout[0] if fresh_last_headout.dim() == 2 else fresh_last_headout  # (vocab_size,)
                init_logits = torch.softmax(headout_1d.unsqueeze(0).unsqueeze(0), dim=-1, dtype=torch.float32)  # (1, 1, vocab_size)
                tree_output = tree.update(init_logits, print_tree=False, tokenizer=tokenizer)
                
                # logits model.model()
                # KV cache 
                past_key_values = fresh_outputs[1]
                init_len = past_key_values[0][0].size(2)
                len_posi = input_ids_for_profile.shape[1] - 1
                logits = fresh_last_headout.unsqueeze(0)
                draft_time = None  # update None
                
                for depth in range(1, max_depth + 1):
                    if tree_output.get("is_final", False):
                        break
                    
                    # update (model_time )
                    # model.model() logits[0] (batch_size, prev_width, vocab_size)
                    logits_for_update = logits.unsqueeze(0) if logits.dim() == 2 else logits  # (prev_width, vocab_size) -> (1, prev_width, vocab_size)
                    tree_output = tree.update(torch.softmax(logits_for_update.to(last_hidden.device), dim=-1, dtype=torch.float32), print_tree=False, tokenizer=tokenizer, draft_time=draft_time)
                    input_ids_step = tree_output["input_ids"].unsqueeze(0)
                    position_ids = tree_output["position_ids"] + len_posi   # [DH] index -> index ( input_ids )
                    if tree_output.get("is_final", False):
                        break
                    
                    # past_key_values
                    if past_key_values is not None:
                        actual_past_length = past_key_values[0][0].shape[2]
                    else:
                        actual_past_length = init_len
                    
                    # tree_attention_mask past_cols
                    tree_attention_mask = tree_output["attention_mask"]
                    current_width = tree_attention_mask.size(0)
                    tree_mask_cols = tree_attention_mask.size(1)
                    past_cols = tree_mask_cols - current_width
                    
                    # process_tree_mask init_len
                    # init_len + past_cols + current_width = actual_past_length + current_width
                    # : init_len = actual_past_length - past_cols
                    init_len_for_mask = actual_past_length - past_cols
                    init_len_for_mask = max(0, init_len_for_mask)
                    
                    tree_attention_mask_with_kv = runner.process_tree_mask(tree_output["attention_mask"], init_len_for_mask)
                    
                    # draft_model.model
                    monitor_started = False
                    if runner.gpu_monitor is not None:
                        try:
                            runner.gpu_monitor.start_monitoring()
                            monitor_started = True
                        except Exception as e:
                            monitor_error_count += 1
                            if not (is_warmup or is_burnin):
                                print(
                                    f"[WARN] GPU monitor start failed during draft profiling "
                                    f"(width={width}, run_idx={run_idx}, depth={depth}): {e}"
                                )
                    if isinstance(input_ids_step, torch.Tensor) and input_ids_step.is_cuda:
                        torch.cuda.synchronize()
                    model_start_time = time.time()
                    draft_outputs = runner.draft_model.model(
                        input_ids=input_ids_step,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        tree_attention_mask=tree_attention_mask_with_kv,
                        return_kv=True,
                        is_draft=True,
                    )
                    if isinstance(draft_outputs[0], torch.Tensor) and draft_outputs[0].is_cuda:
                        torch.cuda.synchronize()
                    draft_time = time.time() - model_start_time
                    if monitor_started and runner.gpu_monitor is not None:
                        try:
                            runner.gpu_monitor.stop_monitoring()
                            _gpu_stats = runner.gpu_monitor.get_stats()
                            _power_w = runner._extract_gpu_power_avg_w(_gpu_stats)
                            _util = _extract_gpu_stat_avg(_gpu_stats, "utilization_percent")
                            _gclk = _extract_gpu_stat_avg(_gpu_stats, "graphics_clock_mhz")
                            _mclk = _extract_gpu_stat_avg(_gpu_stats, "memory_clock_mhz")
                            if not (is_warmup or is_burnin):
                                monitor_sample_total += 1
                                if _power_w is not None:
                                    gpu_power_w_samples.append(float(_power_w))
                                else:
                                    missing_power_count += 1
                                if _util is not None:
                                    gpu_util_samples.append(float(_util))
                                else:
                                    missing_util_count += 1
                                if _gclk is not None:
                                    gpu_graphics_clock_samples.append(float(_gclk))
                                else:
                                    missing_gclk_count += 1
                                if _mclk is not None:
                                    gpu_memory_clock_samples.append(float(_mclk))
                                else:
                                    missing_mclk_count += 1
                        except Exception as e:
                            monitor_error_count += 1
                            if not (is_warmup or is_burnin):
                                print(
                                    f"[WARN] GPU monitor stop/get_stats failed during draft profiling "
                                    f"(width={width}, run_idx={run_idx}, depth={depth}): {e}"
                                )
                    if not (is_warmup or is_burnin):
                        model_call_times.append(draft_time)
                    # Tree KV (local variable)
                    past_key_values = draft_outputs[1]
                    
                    last_hidden = draft_outputs[0]
                    last_headout = runner.draft_model.lm_head(last_hidden)
                    logits = last_headout
                
                for _var in ["past_key_values", "draft_outputs", "tree_output", "logits_for_update", "tree_attention_mask_with_kv", "input_ids_step", "position_ids"]:
                    if _var in locals():
                        del locals()[_var]
                torch.cuda.empty_cache()
                run_idx += 1
                if (not is_warmup) and (not is_burnin) and (len(model_call_times) >= target_model_calls):
                    break
                if run_idx >= max_total_runs:
                    raise RuntimeError(
                        f"Draft profiling reached run safety limit before hitting target model calls "
                        f"(width={width}, target_calls={target_model_calls}, collected_calls={len(model_call_times)}, "
                        f"max_total_runs={max_total_runs})."
                    )
        
            # draft_model.model 
            count_call = len(model_call_times)
            total_ms_call = sum(model_call_times) * 1000 if count_call > 0 else 0.0
            avg_ms_call = (total_ms_call / count_call) if count_call > 0 else 0.0
            min_ms_call = (min(model_call_times) * 1000) if count_call > 0 else 0.0
            max_ms_call = (max(model_call_times) * 1000) if count_call > 0 else 0.0
            gpu_power_avg_w = (
                float(sum(gpu_power_w_samples) / len(gpu_power_w_samples))
                if gpu_power_w_samples
                else 0.0
            )
            gpu_util_avg_percent = (
                float(sum(gpu_util_samples) / len(gpu_util_samples))
                if gpu_util_samples
                else 0.0
            )
            gpu_graphics_clock_avg_mhz = (
                float(sum(gpu_graphics_clock_samples) / len(gpu_graphics_clock_samples))
                if gpu_graphics_clock_samples
                else 0.0
            )
            gpu_memory_clock_avg_mhz = (
                float(sum(gpu_memory_clock_samples) / len(gpu_memory_clock_samples))
                if gpu_memory_clock_samples
                else 0.0
            )
            gpu_total_energy_kwh = (
                (gpu_power_avg_w * float(sum(model_call_times))) / 3600000.0
                if count_call > 0 and gpu_power_avg_w > 0
                else 0.0
            )
            gpu_energy_per_call_kwh = (
                (gpu_total_energy_kwh / float(count_call))
                if count_call > 0 and gpu_total_energy_kwh > 0
                else 0.0
            )
            draft_cost_per_call_usd = (
                gpu_energy_per_call_kwh * float(runner.draft_electricity_cost_per_kwh)
                if gpu_energy_per_call_kwh > 0
                else 0.0
            )
            if require_gpu_energy_profile and gpu_power_avg_w <= 0:
                print(
                    f"[WARN] {runner.objective_metric} profiling without valid power samples "
                    f"(width={width}): draft_cost_per_call_usd will be 0 for this width."
                )

            power_valid_count = len(gpu_power_w_samples)
            util_valid_count = len(gpu_util_samples)
            gclk_valid_count = len(gpu_graphics_clock_samples)
            mclk_valid_count = len(gpu_memory_clock_samples)
            total_monitor = int(monitor_sample_total)

            def _ratio_pct(missing_count: int, total_count: int) -> float:
                return float(missing_count * 100.0 / total_count) if total_count > 0 else 0.0
            
            profile_data[width] = {
                'model_call_count': count_call,
                'model_call_total_time_ms': total_ms_call,
                'model_call_avg_time_ms': avg_ms_call,
                'model_call_min_time_ms': min_ms_call,
                'model_call_max_time_ms': max_ms_call,
                'gpu_power_avg_w': gpu_power_avg_w,
                'gpu_util_avg_percent': gpu_util_avg_percent,
                'gpu_graphics_clock_avg_mhz': gpu_graphics_clock_avg_mhz,
                'gpu_memory_clock_avg_mhz': gpu_memory_clock_avg_mhz,
                'gpu_monitor_sample_count': total_monitor,
                'gpu_monitor_error_count': int(monitor_error_count),
                'gpu_power_sample_valid_count': int(power_valid_count),
                'gpu_power_sample_missing_count': int(max(0, total_monitor - power_valid_count)),
                'gpu_power_sample_missing_percent': _ratio_pct(max(0, total_monitor - power_valid_count), total_monitor),
                'gpu_util_sample_valid_count': int(util_valid_count),
                'gpu_util_sample_missing_count': int(max(0, total_monitor - util_valid_count)),
                'gpu_util_sample_missing_percent': _ratio_pct(max(0, total_monitor - util_valid_count), total_monitor),
                'gpu_graphics_clock_sample_valid_count': int(gclk_valid_count),
                'gpu_graphics_clock_sample_missing_count': int(max(0, total_monitor - gclk_valid_count)),
                'gpu_graphics_clock_sample_missing_percent': _ratio_pct(max(0, total_monitor - gclk_valid_count), total_monitor),
                'gpu_memory_clock_sample_valid_count': int(mclk_valid_count),
                'gpu_memory_clock_sample_missing_count': int(max(0, total_monitor - mclk_valid_count)),
                'gpu_memory_clock_sample_missing_percent': _ratio_pct(max(0, total_monitor - mclk_valid_count), total_monitor),
                'gpu_total_energy_kwh': gpu_total_energy_kwh,
                'gpu_energy_per_call_kwh': gpu_energy_per_call_kwh,
                'draft_cost_per_call_usd': draft_cost_per_call_usd,
            }
            if total_monitor > 0 and (
                missing_power_count > 0
                or missing_util_count > 0
                or missing_gclk_count > 0
                or missing_mclk_count > 0
                or monitor_error_count > 0
            ):
                print(
                    f"[WARN] width={width} GPU metric missing summary: "
                    f"power {max(0, total_monitor - power_valid_count)}/{total_monitor} "
                    f"({ _ratio_pct(max(0, total_monitor - power_valid_count), total_monitor):.2f}%), "
                    f"util {max(0, total_monitor - util_valid_count)}/{total_monitor} "
                    f"({ _ratio_pct(max(0, total_monitor - util_valid_count), total_monitor):.2f}%), "
                    f"gclk {max(0, total_monitor - gclk_valid_count)}/{total_monitor} "
                    f"({ _ratio_pct(max(0, total_monitor - gclk_valid_count), total_monitor):.2f}%), "
                    f"mclk {max(0, total_monitor - mclk_valid_count)}/{total_monitor} "
                    f"({ _ratio_pct(max(0, total_monitor - mclk_valid_count), total_monitor):.2f}%), "
                    f"monitor_errors={monitor_error_count}"
                )
            print(
                f"  width={width}: model calls {count_call} calls, total {total_ms_call:.3f}ms, "
                f"average {avg_ms_call:.3f}ms, minimum {min_ms_call:.3f}ms, maximum {max_ms_call:.3f}ms, "
                f"gpu_power_avg={gpu_power_avg_w:.3f}W, gpu_util_avg={gpu_util_avg_percent:.2f}%, "
                f"gclk_avg={gpu_graphics_clock_avg_mhz:.1f}MHz, mclk_avg={gpu_memory_clock_avg_mhz:.1f}MHz, "
                f"gpu_total_energy={gpu_total_energy_kwh:.9f}kWh"
            )
    
    # JSON
    if profile_data:
        with open(profile_file, 'w') as f:
            json.dump(profile_data, f, indent=2)
        print(f"\nProfiling data saved: {profile_file}")
        print(f"Collected width count: {len(profile_data)}")
    else:
        print("\nNo profiling data available.")
        return None
    
    return profile_data


def run_warmup(
    sock,
    runner: DraftRunner,
    tokenizer: AutoTokenizer,
    questions: List[dict],
    base_model_path: str,
    bench_name: str,
    nodes: int,
    max_depth: int,
    per_token_probability_bound: float,
    per_path_probability_bound: float,
    min_width: int,
    fixed_width: bool,
    fixed_width_value: int,
    fixed_nnodes: bool,
    fixed_depth: bool,
    debug: bool,
    warmup_cost_sensitivity: float = None,
    warmup_rounds: int = 3,
    full_query: bool = False,
    cs_anchor_cycle: List[float] = None,
    max_steps_per_round: Optional[int] = None,
    select_last_trial: bool = False,
    fixed_accept_length_scale: Optional[float] = None,
    update_accept_length_ratio: bool = True,
    preserve_accept_length_ratio_stats: bool = False,
    reference_token_count_mode: str = "actual",
    diverse_questions_per_round: bool = False,
    question_start_index: int = 0,
    aggregate_round_metrics_mean: bool = False,
):
    """Warmup/reference execution common function.
    - full_query=False: short warmup (same early-stop condition as before)
    - full_query=True : process one question as far as possible (for reference calculation)
    - diverse_questions_per_round=True increments the question index each round when enabled
    """
    if not questions:
        return {"token_per_second": 0.0, "cost_per_token": 0.0, "objective_per_token": 0.0}
    original_cost_sensitivity = runner.cost_sensitivity
    if warmup_cost_sensitivity is not None:
        runner.cost_sensitivity = float(warmup_cost_sensitivity)
    warmup_total_new_tokens_actual = 0.0
    warmup_total_new_tokens_for_metric = 0.0
    warmup_total_wall_time_sec = 0.0
    warmup_total_draft_time_sec = 0.0
    warmup_total_target_verification_time_sec = 0.0
    warmup_total_d2t_bytes = 0.0
    warmup_total_t2d_bytes = 0.0
    warmup_step_metric_per_token = []
    warmup_step_tps = []
    per_cs_step_metric = {}
    per_cs_step_tps = {}
    per_cs_trace_steps = {}
    last_trial_tps = None
    last_trial_cost_per_token = None
    last_trial_objective_per_token = None
    last_trial_actual_accept_length = None
    last_trial_effective_accept_length_for_metric = None
    last_trial_expected_accept_raw = None
    last_trial_expected_accept_scaled = None
    last_trial_expected_accept_clipped = None
    original_scale_override = getattr(runner, "accept_length_scale_override", None)
    token_count_mode = str(reference_token_count_mode or "actual").strip().lower()
    if token_count_mode not in ("actual", "clipped_expected"):
        token_count_mode = "actual"
    if fixed_accept_length_scale is not None:
        runner.accept_length_scale_override = float(fixed_accept_length_scale)
    try:
        run_rounds = max(1, int(warmup_rounds))
        question_count = max(1, int(len(questions)))
        start_idx = int(question_start_index) if question_start_index is not None else 0
        tested_question_indices = []
        round_tps_values = []
        round_cost_per_token_values = []
        round_objective_per_token_values = []
        for round_idx in range(run_rounds):
            torch.manual_seed(0)
            # Warmup KV cache
            runner.reset_kv()
            if bool(diverse_questions_per_round):
                q_idx = (start_idx + int(round_idx)) % question_count
            else:
                q_idx = 0
            tested_question_indices.append(int(q_idx))
            warm_q = questions[q_idx]
            conv = _build_conversation_template_for_model(base_model_path)
            warm_turns = _extract_question_turns(warm_q, bench_name)
            conv.append_message(conv.roles[0], warm_turns[0])
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt() + " "
            input_ids = tokenizer([prompt]).input_ids
            input_ids_t = torch.as_tensor(input_ids).to("cuda")
            turn_wall_start_time = time.time()

            send_json_with_size(sock, {"type": "init", "input_ids": input_ids[0]})
            reply, _ = recv_json_with_size(sock)
            if reply.get("type") != "init_ok":
                break
            current_next_token = reply["next_token"]

            warm_steps = 0
            round_total_new_tokens_for_metric = 0.0
            round_total_cost = 0.0
            round_total_objective = 0.0
            while True:
                current_step_cs = None
                if cs_anchor_cycle:
                    try:
                        current_step_cs = float(cs_anchor_cycle[warm_steps % len(cs_anchor_cycle)])
                        runner.cost_sensitivity = float(current_step_cs)
                    except Exception:
                        current_step_cs = None
                tree_build_start_time = time.time()
                if runner.gpu_monitor:
                    runner.gpu_monitor.start_monitoring()
                draft_ids, draft_pos, tree_mask, parent, tree_depth, final_nnodes, depth_widths, node_meta = build_tree_with_next_token(
                    runner, input_ids_t, nodes, max_depth, current_next_token, tokenizer, debug, print_tree=False, per_token_probability_bound=per_token_probability_bound, per_path_probability_bound=per_path_probability_bound, min_width=min_width, fixed_width=fixed_width, fixed_width_value=fixed_width_value, fixed_nnodes=fixed_nnodes, fixed_depth=fixed_depth
                )
                tree_build_end_time = time.time()
                payload = {
                    "type": "tree_step",
                    "draft_input_ids": draft_ids[0].tolist(),
                    "draft_position_ids": draft_pos.tolist(),
                    "tree_attention_mask": tree_mask.tolist(),
                    "parent": parent.tolist(),
                }
                # Draft Target : Draft
                send_start_time = time.time()
                d2t_bytes = send_json_with_size(sock, payload)
                # + GPU
                if runner.gpu_monitor:
                    runner.gpu_monitor.stop_monitoring()
                    warmup_gpu_stats = runner.gpu_monitor.get_stats()
                    runner.update_draft_objective_rate_from_gpu(
                        warmup_gpu_stats,
                        require_valid=bool(runner.uses_total_cost_objective() and not runner.no_draft_cost),
                    )

                # Target Draft : Draft Target
                reply, t2d_bytes = recv_json_with_size(sock)
                recv_end_time = time.time()
                warmup_total_d2t_bytes += float(d2t_bytes)
                warmup_total_t2d_bytes += float(t2d_bytes)
                if reply.get("type") != "verify_result":
                    break
                accepted_tokens: List[int] = reply["accepted_tokens"]
                accept_length = int(len(accepted_tokens))
                current_next_token = reply["next_token"]
                eos_reached: bool = reply["eos_reached"]
                best_ids: List[int] = reply["best_ids"]
                base_input_len: int = reply.get("base_input_len", input_ids_t.shape[1])
                target_verification_time = reply.get("target_verification_time_ms", None)
                target_energy_rate_per_sec = reply.get("target_energy_rate_per_sec", None)
                if target_verification_time is not None:
                    warmup_total_target_verification_time_sec += max(0.0, float(target_verification_time) / 1000.0)
                runner.update_target_objective_rate(target_energy_rate_per_sec)

                # TODO:Accept KV
                # base_kv_len draft() input_ids (old_input_ids)
                # draft_stable_kv draft() ,
                # base_kv_len draft_stable_kv
                # base_kv_len = base_input_len
                # update_kv_with_accepted(runner, best_ids, base_kv_len, current_next_token, debug)
                # print(runner.draft_stable_kv[0][0].shape[2])

                input_ids_t = torch.cat([
                    input_ids_t,
                    torch.tensor([accepted_tokens], device=input_ids_t.device, dtype=torch.long),
                ], dim=-1)
                warmup_total_new_tokens_actual += float(len(accepted_tokens) + 1)  # accepted + next_token

                # tree (warmup )
                # timing_stats total_time_seconds ,
                timing_stats = reply.get("timing_stats", {})
                # total_time = timing_stats.get("total_time_seconds", None)
                # if total_time is None:
                # E2E :
                # tree build + Draft Target Draft
                tree_build_time_sec = max(0.0, float(tree_build_end_time - tree_build_start_time))
                network_roundtrip_sec = max(0.0, float(recv_end_time - send_start_time))
                total_time = tree_build_time_sec + network_roundtrip_sec
                warmup_total_draft_time_sec += float(tree_build_time_sec)

                # (warmup )
                # warmup None
                transfer_time = None
                draft_to_target_time = 0.0
                target_to_draft_time = 0.0
                target_recv_end_time = reply.get("target_recv_end_time", None)
                target_send_start_time = reply.get("target_send_start_time", None)
                if target_recv_end_time is not None and target_send_start_time is not None:
                    draft_to_target_time = max(0.05, (target_recv_end_time - send_start_time) * 1000.0)  # ms
                    target_to_draft_time = max(0.05, (recv_end_time - target_send_start_time) * 1000.0)  # ms
                    transfer_time = (draft_to_target_time + target_to_draft_time) / 1000.0

                runner.prev_tree_final_nnodes = final_nnodes
                runner.prev_tree_depth = tree_depth
                runner.prev_tree_total_target_time = total_time
                runner.prev_tree_transfer_time = transfer_time
                runner.prev_tree_accept_length = int(accept_length)
                runner.per_token_draft_to_target_transfer_time = (draft_to_target_time / 1000.0) / final_nnodes if final_nnodes > 0 else 0.0
                runner.per_token_target_to_draft_transfer_time = (target_to_draft_time / 1000.0) / len(accepted_tokens) if len(accepted_tokens) > 0 else 0.0
                runner.per_token_draft_to_target_bytes = (float(d2t_bytes) / final_nnodes) if final_nnodes > 0 else 0.0
                runner.per_token_target_to_draft_bytes = (float(t2d_bytes) / len(accepted_tokens)) if len(accepted_tokens) > 0 else 0.0
                step_bytes_total = max(0.0, float(d2t_bytes) + float(t2d_bytes))
                step_total_time_sec = max(1e-9, float(tree_build_time_sec + network_roundtrip_sec))
                step_target_cost = max(0.0, float(target_verification_time or 0.0) / 1000.0) * float(runner.target_per_sec_cost)
                step_inbound_comm_cost = (
                    (max(0.0, float(d2t_bytes)) / float(1024 ** 3))
                    * float(runner.user_communication_cost_per_gb)
                )
                step_outbound_comm_cost = (
                    (max(0.0, float(t2d_bytes)) / float(1024 ** 3))
                    * float(
                        runner.user_communication_cost_per_gb
                        + runner.cloud_outbound_cost_per_gb
                    )
                )
                step_draft_rate_for_cost = (
                    float(runner.get_draft_objective_rate_per_sec())
                    if runner.uses_total_cost_objective()
                    else float(getattr(runner, "draft_per_sec_cost", 0.0))
                )
                step_draft_cost = (
                    0.0
                    if bool(getattr(runner, "no_draft_cost", False))
                    else max(0.0, float(tree_build_time_sec)) * max(0.0, step_draft_rate_for_cost)
                )
                step_cost_total = (
                    step_draft_cost
                    + step_target_cost
                    + max(0.0, float(step_inbound_comm_cost))
                    + max(0.0, float(step_outbound_comm_cost))
                )
                if not runner.uses_any_cost_objective():
                    step_objective_total = 0.0
                    include_draft_energy = (
                        runner.uses_draft_energy_objective()
                        or runner.uses_server_only_target_energy_sum()
                    )
                    if include_draft_energy:
                        step_objective_total += max(
                            0.0, float(runner.get_draft_objective_rate_per_sec()) * float(tree_build_time_sec)
                        )
                    if runner.uses_target_energy_objective():
                        step_objective_total += max(
                            0.0,
                            float(runner.get_target_objective_rate_per_sec())
                            * max(0.0, float(target_verification_time or 0.0) / 1000.0),
                        )
                else:
                    if runner.uses_api_cost_objective():
                        step_objective_total = step_target_cost + max(0.0, float(step_outbound_comm_cost))
                    else:
                        step_objective_total = step_cost_total
                exp_raw = None
                exp_scaled = None
                exp_clipped = None
                try:
                    exp_raw_candidate = (
                        float(runner.last_sum_expected_accepted_length)
                        if runner.last_sum_expected_accepted_length is not None
                        else None
                    )
                    if exp_raw_candidate is not None and np.isfinite(exp_raw_candidate):
                        exp_raw = exp_raw_candidate
                        scale_used = float(getattr(runner, "last_accept_length_scale_used", 1.0))
                        if np.isfinite(scale_used):
                            exp_scaled = exp_raw * scale_used
                            exp_clipped = exp_scaled * (1.0 - float(getattr(runner, "accept_length_margin", 0.05)))
                except Exception:
                    exp_raw = None
                    exp_scaled = None
                    exp_clipped = None

                effective_accept_length_for_metric = float(accept_length)
                if token_count_mode == "clipped_expected":
                    if exp_clipped is not None and np.isfinite(exp_clipped):
                        effective_accept_length_for_metric = max(0.0, float(exp_clipped))
                step_new_tokens_for_metric = max(1e-9, float(effective_accept_length_for_metric + 1.0))
                warmup_total_new_tokens_for_metric += float(step_new_tokens_for_metric)
                round_total_new_tokens_for_metric += float(step_new_tokens_for_metric)

                step_objective_per_token = float(step_objective_total / step_new_tokens_for_metric)
                step_cost_per_token = float(step_cost_total / step_new_tokens_for_metric)
                step_tps = float(step_new_tokens_for_metric / step_total_time_sec)
                round_total_objective += float(step_objective_total)
                round_total_cost += float(step_cost_total)
                warmup_step_metric_per_token.append(step_objective_per_token)
                warmup_step_tps.append(step_tps)
                if current_step_cs is not None:
                    cs_key = f"{float(current_step_cs):.6g}"
                    per_cs_step_metric.setdefault(cs_key, []).append(step_objective_per_token)
                    per_cs_step_tps.setdefault(cs_key, []).append(step_tps)
                else:
                    cs_key = f"{float(runner.cost_sensitivity):.6g}"
                depth_widths_serialized = []
                try:
                    if isinstance(depth_widths, list):
                        depth_widths_serialized = [int(x) for x in depth_widths]
                except Exception:
                    depth_widths_serialized = []
                per_cs_trace_steps.setdefault(cs_key, []).append({
                    "round_idx": int(round_idx),
                    "question_index": int(q_idx),
                    "step_index": int(warm_steps),
                    "cost_sensitivity": float(current_step_cs if current_step_cs is not None else runner.cost_sensitivity),
                    "tree_depth": int(tree_depth),
                    "final_nnodes": int(final_nnodes),
                    "depth_widths": depth_widths_serialized,
                    "accept_length": int(accept_length),
                    "effective_accept_length_for_metric": float(effective_accept_length_for_metric),
                    "step_tps": float(step_tps),
                    "step_metric_per_token": float(step_objective_per_token),
                    "step_metric_per_1m_token": float(step_objective_per_token * 1_000_000.0),
                    "step_cost_per_token": float(step_cost_per_token),
                    "step_cost_per_1m_token": float(step_cost_per_token * 1_000_000.0),
                    "tree_build_time_sec": float(tree_build_time_sec),
                    "network_roundtrip_sec": float(network_roundtrip_sec),
                    "target_verification_time_sec": float(max(0.0, float(target_verification_time or 0.0) / 1000.0)),
                    "step_cost_components": {
                        "draft_cost": float(step_draft_cost),
                        "target_cost": float(step_target_cost),
                        "inbound_comm_cost": float(step_inbound_comm_cost),
                        "outbound_comm_cost": float(step_outbound_comm_cost),
                        "total_cost": float(step_cost_total),
                    },
                    "step_objective_total": float(step_objective_total),
                })

                if bool(update_accept_length_ratio) and exp_raw is not None and exp_raw > 0 and accept_length > 0:
                    runner.accept_length_actual_sum += float(accept_length)
                    runner.accept_length_expected_sum += float(exp_raw)

                last_trial_tps = step_tps
                last_trial_cost_per_token = step_cost_per_token
                last_trial_objective_per_token = step_objective_per_token
                last_trial_actual_accept_length = int(accept_length)
                last_trial_effective_accept_length_for_metric = float(effective_accept_length_for_metric)
                last_trial_expected_accept_raw = exp_raw
                last_trial_expected_accept_scaled = exp_scaled
                last_trial_expected_accept_clipped = exp_clipped

                warm_steps += 1
                reached_step_limit = (
                    max_steps_per_round is not None
                    and int(max_steps_per_round) > 0
                    and warm_steps >= int(max_steps_per_round)
                )
                if full_query:
                    # [MODIFIED] Use model's max_position_embeddings minus tree buffer to avoid
                    # KV cache overflow on short-context models (e.g. LLaMA-2 4096).
                    _max_pos = getattr(runner.draft_model.config, "max_position_embeddings", 4096)
                    _safe_limit = max(512, _max_pos - nodes * max_depth - 64)
                    should_stop = reached_step_limit or eos_reached or input_ids_t.shape[1] > _safe_limit
                else:
                    should_stop = reached_step_limit or eos_reached or warm_steps > 5 or input_ids_t.shape[1] > 512
                if should_stop:
                    break
            round_wall_time_sec = max(0.0, float(time.time() - turn_wall_start_time))
            warmup_total_wall_time_sec += float(round_wall_time_sec)
            if round_total_new_tokens_for_metric > 0 and round_wall_time_sec > 0:
                round_tps = float(round_total_new_tokens_for_metric / round_wall_time_sec)
                round_tps_values.append(float(round_tps))
                round_cost_per_token = float(round_total_cost / round_total_new_tokens_for_metric)
                round_cost_per_token_values.append(float(round_cost_per_token))
                if round_total_objective > 0:
                    round_obj_per_token = float(round_total_objective / round_total_new_tokens_for_metric)
                elif round_tps > 0:
                    round_obj_per_token = float(1.0 / round_tps)
                else:
                    round_obj_per_token = 0.0
                round_objective_per_token_values.append(float(round_obj_per_token))

        # Warmup : warmup (run)
        # calibration accept_length .
        _accept_actual_sum_before_reset = float(getattr(runner, "accept_length_actual_sum", 0.0))
        _accept_expected_sum_before_reset = float(getattr(runner, "accept_length_expected_sum", 0.0))
        runner.reset_timing_stats(reset_global=True)
        if bool(preserve_accept_length_ratio_stats):
            runner.accept_length_actual_sum = _accept_actual_sum_before_reset
            runner.accept_length_expected_sum = _accept_expected_sum_before_reset
        print("Warmup done")
        warmup_tps = (
            warmup_total_new_tokens_for_metric / warmup_total_wall_time_sec
            if warmup_total_wall_time_sec > 0
            else 0.0
        )
        bytes_per_gb = float(1024 ** 3)
        warmup_inbound_cost = (
            (warmup_total_d2t_bytes / bytes_per_gb) * float(runner.user_communication_cost_per_gb)
            if bytes_per_gb > 0
            else 0.0
        )
        warmup_outbound_cost = (
            (warmup_total_t2d_bytes / bytes_per_gb)
            * float(runner.user_communication_cost_per_gb + runner.cloud_outbound_cost_per_gb)
            if bytes_per_gb > 0
            else 0.0
        )
        warmup_comm_cost = warmup_inbound_cost + warmup_outbound_cost
        warmup_draft_rate_for_cost = (
            float(runner.get_draft_objective_rate_per_sec())
            if runner.uses_total_cost_objective()
            else float(getattr(runner, "draft_per_sec_cost", 0.0))
        )
        warmup_draft_cost = 0.0 if runner.no_draft_cost else (
            warmup_total_draft_time_sec * max(0.0, warmup_draft_rate_for_cost)
        )
        warmup_target_cost = warmup_total_target_verification_time_sec * float(runner.target_per_sec_cost)
        warmup_total_cost = warmup_draft_cost + warmup_target_cost + warmup_comm_cost
        warmup_cost_per_token = (
            warmup_total_cost / warmup_total_new_tokens_for_metric
            if warmup_total_new_tokens_for_metric > 0
            else 0.0
        )
        if not runner.uses_any_cost_objective():
            warmup_energy_objective = 0.0
            if runner.uses_draft_energy_objective() or runner.uses_server_only_target_energy_sum():
                warmup_energy_objective += (
                    max(0.0, warmup_total_draft_time_sec) * float(runner.get_draft_objective_rate_per_sec())
                )
            if runner.uses_target_energy_objective():
                warmup_energy_objective += (
                    max(0.0, warmup_total_target_verification_time_sec)
                    * float(runner.get_target_objective_rate_per_sec())
                )
            warmup_objective_per_token = (
                warmup_energy_objective / warmup_total_new_tokens_for_metric
                if warmup_total_new_tokens_for_metric > 0
                else 0.0
            )
        else:
            warmup_objective_total_cost = warmup_total_cost
            if runner.uses_api_cost_objective():
                warmup_objective_total_cost = warmup_target_cost + warmup_outbound_cost
            warmup_objective_per_token = (
                warmup_objective_total_cost / warmup_total_new_tokens_for_metric
                if warmup_total_new_tokens_for_metric > 0
                else 0.0
            )
        if warmup_objective_per_token <= 0 and warmup_tps > 0:
            # objective rate fallback
            warmup_objective_per_token = 1.0 / warmup_tps
        feasible_metric_per_token = {
            "min": float(min(warmup_step_metric_per_token)) if warmup_step_metric_per_token else None,
            "max": float(max(warmup_step_metric_per_token)) if warmup_step_metric_per_token else None,
            "mean": (
                float(sum(warmup_step_metric_per_token) / len(warmup_step_metric_per_token))
                if warmup_step_metric_per_token
                else None
            ),
            "count": int(len(warmup_step_metric_per_token)),
        }
        feasible_tps = {
            "min": float(min(warmup_step_tps)) if warmup_step_tps else None,
            "max": float(max(warmup_step_tps)) if warmup_step_tps else None,
            "mean": (
                float(sum(warmup_step_tps) / len(warmup_step_tps))
                if warmup_step_tps
                else None
            ),
            "count": int(len(warmup_step_tps)),
        }
        cs_anchor_curve = []
        if per_cs_step_metric:
            for cs_key in sorted(per_cs_step_metric.keys(), key=lambda x: float(x)):
                m_vals = per_cs_step_metric.get(cs_key, [])
                t_vals = per_cs_step_tps.get(cs_key, [])
                if not m_vals or not t_vals:
                    continue
                metric_mean = float(sum(m_vals) / len(m_vals))
                tps_mean = float(sum(t_vals) / len(t_vals))
                cs_anchor_curve.append({
                    "cost_sensitivity": float(cs_key),
                    "predicted_tps": tps_mean,
                    "predicted_metric_per_token": metric_mean,
                    "predicted_metric_per_1m_token": metric_mean * 1_000_000.0,
                    "count": int(min(len(m_vals), len(t_vals))),
                })

        # Feasible range endpoints: cs=0 and cs=1.
        if cs_anchor_cycle and cs_anchor_curve:
            cs_map = {round(float(row.get("cost_sensitivity", -1.0)), 6): row for row in cs_anchor_curve}
            row0 = cs_map.get(0.0)
            row1 = cs_map.get(1.0)
            if row0 is not None and row1 is not None:
                m0 = float(row0.get("predicted_metric_per_token", 0.0))
                m1 = float(row1.get("predicted_metric_per_token", 0.0))
                t0 = float(row0.get("predicted_tps", 0.0))
                t1 = float(row1.get("predicted_tps", 0.0))
                feasible_metric_per_token = {
                    "min": float(min(m0, m1)),
                    "max": float(max(m0, m1)),
                    "mean": float((m0 + m1) / 2.0),
                    "count": 2,
                }
                feasible_tps = {
                    "min": float(min(t0, t1)),
                    "max": float(max(t0, t1)),
                    "mean": float((t0 + t1) / 2.0),
                    "count": 2,
                }
        by_cs_summary = {}
        for cs_key, trace_rows in per_cs_trace_steps.items():
            if not trace_rows:
                continue
            draft_cost_mean = float(
                sum(float(x.get("step_cost_components", {}).get("draft_cost", 0.0)) for x in trace_rows) / len(trace_rows)
            )
            target_cost_mean = float(
                sum(float(x.get("step_cost_components", {}).get("target_cost", 0.0)) for x in trace_rows) / len(trace_rows)
            )
            comm_cost_mean = float(
                sum(
                    float(x.get("step_cost_components", {}).get("inbound_comm_cost", 0.0))
                    + float(x.get("step_cost_components", {}).get("outbound_comm_cost", 0.0))
                    for x in trace_rows
                ) / len(trace_rows)
            )
            component_map = {
                "draft_cost": draft_cost_mean,
                "target_cost": target_cost_mean,
                "comm_cost": comm_cost_mean,
            }
            dominant_component = max(component_map.items(), key=lambda kv: float(kv[1]))[0]
            avg_width_vals = []
            for row in trace_rows:
                dws = row.get("depth_widths", [])
                if isinstance(dws, list) and dws:
                    avg_width_vals.append(float(sum(float(v) for v in dws) / len(dws)))
            by_cs_summary[str(cs_key)] = {
                "step_count": int(len(trace_rows)),
                "avg_tree_depth": float(sum(float(x.get("tree_depth", 0.0)) for x in trace_rows) / len(trace_rows)),
                "avg_final_nnodes": float(sum(float(x.get("final_nnodes", 0.0)) for x in trace_rows) / len(trace_rows)),
                "avg_width_over_depths": (
                    float(sum(avg_width_vals) / len(avg_width_vals)) if avg_width_vals else None
                ),
                "avg_cost_components": {
                    "draft_cost": draft_cost_mean,
                    "target_cost": target_cost_mean,
                    "comm_cost": comm_cost_mean,
                },
                "dominant_cost_component": str(dominant_component),
            }
        reference_debug_trace = {
            "token_count_mode": str(token_count_mode),
            "tested_question_indices": [int(x) for x in tested_question_indices],
            "by_cs_summary": by_cs_summary,
            "step_records": per_cs_trace_steps,
        }
        output_tps = float(warmup_tps)
        output_cost_per_token = float(warmup_cost_per_token)
        output_objective_per_token = float(warmup_objective_per_token)
        if bool(select_last_trial) and last_trial_tps is not None:
            output_tps = float(last_trial_tps)
            output_cost_per_token = float(last_trial_cost_per_token) if last_trial_cost_per_token is not None else output_cost_per_token
            output_objective_per_token = float(last_trial_objective_per_token) if last_trial_objective_per_token is not None else output_objective_per_token
            if output_objective_per_token <= 0 and output_tps > 0:
                output_objective_per_token = 1.0 / output_tps
            feasible_metric_per_token = {
                "min": float(output_objective_per_token),
                "max": float(output_objective_per_token),
                "mean": float(output_objective_per_token),
                "count": 1,
            }
            feasible_tps = {
                "min": float(output_tps),
                "max": float(output_tps),
                "mean": float(output_tps),
                "count": 1,
            }
        elif bool(aggregate_round_metrics_mean) and round_tps_values:
            output_tps = float(sum(round_tps_values) / len(round_tps_values))
            if round_cost_per_token_values:
                output_cost_per_token = float(
                    sum(round_cost_per_token_values) / len(round_cost_per_token_values)
                )
            if round_objective_per_token_values:
                output_objective_per_token = float(
                    sum(round_objective_per_token_values) / len(round_objective_per_token_values)
                )
            if output_objective_per_token <= 0 and output_tps > 0:
                output_objective_per_token = float(1.0 / output_tps)

        accept_length_error_last = None
        accept_length_error_ratio_last = None
        if (
            last_trial_actual_accept_length is not None
            and last_trial_expected_accept_raw is not None
            and float(last_trial_expected_accept_raw) > 0
        ):
            accept_length_error_last = float(last_trial_actual_accept_length) - float(last_trial_expected_accept_raw)
            accept_length_error_ratio_last = float(last_trial_actual_accept_length) / float(last_trial_expected_accept_raw)

        return {
            "token_per_second": float(output_tps),
            "cost_per_token": float(output_cost_per_token),
            "objective_per_token": float(output_objective_per_token),
            "draft_objective_rate_per_sec": float(runner.get_draft_objective_rate_per_sec()),
            "target_objective_rate_per_sec": float(runner.get_target_objective_rate_per_sec()),
            "feasible_metric_per_token": feasible_metric_per_token,
            "feasible_tps": feasible_tps,
            "reference_cs_anchor_curve": cs_anchor_curve,
            "trial_repeat_count": int(run_rounds),
            "trial_selection_rule": "last_trial" if bool(select_last_trial) else "mean",
            "actual_accept_length_last": (
                int(last_trial_actual_accept_length)
                if last_trial_actual_accept_length is not None
                else None
            ),
            "expected_accept_length_raw_last": (
                float(last_trial_expected_accept_raw)
                if last_trial_expected_accept_raw is not None
                else None
            ),
            "expected_accept_length_scaled_last": (
                float(last_trial_expected_accept_scaled)
                if last_trial_expected_accept_scaled is not None
                else None
            ),
            "expected_accept_length_clipped_last": (
                float(last_trial_expected_accept_clipped)
                if last_trial_expected_accept_clipped is not None
                else None
            ),
            "accept_length_error_last": (
                float(accept_length_error_last)
                if accept_length_error_last is not None
                else None
            ),
            "accept_length_error_ratio_last": (
                float(accept_length_error_ratio_last)
                if accept_length_error_ratio_last is not None
                else None
            ),
            "reference_token_count_mode": str(token_count_mode),
            "total_new_tokens_actual": float(warmup_total_new_tokens_actual),
            "total_new_tokens_for_metric": float(warmup_total_new_tokens_for_metric),
            "effective_accept_length_for_metric_last": (
                float(last_trial_effective_accept_length_for_metric)
                if last_trial_effective_accept_length_for_metric is not None
                else None
            ),
            "tested_question_indices": [int(x) for x in tested_question_indices],
            "round_tps_mean": (
                float(sum(round_tps_values) / len(round_tps_values))
                if round_tps_values
                else None
            ),
            "round_cost_per_token_mean": (
                float(sum(round_cost_per_token_values) / len(round_cost_per_token_values))
                if round_cost_per_token_values
                else None
            ),
            "round_objective_per_token_mean": (
                float(sum(round_objective_per_token_values) / len(round_objective_per_token_values))
                if round_objective_per_token_values
                else None
            ),
            "reference_debug_trace": reference_debug_trace,
        }
    finally:
        runner.accept_length_scale_override = original_scale_override
        if runner.gpu_monitor:
            try:
                runner.gpu_monitor.stop_monitoring()
            except Exception:
                pass
        runner.cost_sensitivity = original_cost_sensitivity


def _sanitize_key_component(value: str, max_len: int = 48) -> str:
    if value is None:
        value = "none"
    value = str(value).strip().lower()
    safe = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("-")
    normalized = "".join(safe)
    while "--" in normalized:
        normalized = normalized.replace("--", "-")
    normalized = normalized.strip("-") or "none"
    return normalized[:max_len]


def _normalize_model_identifier(value: str) -> str:
    """Normalize model identification strings for cache keys."""
    if value is None:
        return "unknown"
    normalized = str(value).strip().replace("\\", "/").rstrip("/")
    normalized = normalized.lower()
    return normalized or "unknown"


def _reference_cache_paths(
    base_model_path: str,
    draft_model_path: str,
    bench_name: str,
    objective_metric: str,
    server_name: str,
    device_name: str,
    target_quantization: str = "8bit",
    draft_quantization: str = "8bit",
    objective_selection_mode: str = None,
    reference_mode_key: str = None,
):
    script_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(script_dir)
    reference_dir = os.path.join(_resolve_data_root(parent_dir), "reference")
    os.makedirs(reference_dir, exist_ok=True)

    normalized_target_model = _normalize_model_identifier(base_model_path)
    normalized_draft_model = _normalize_model_identifier(draft_model_path)
    key_payload = {
        "server_gpu": str(server_name) if server_name is not None else "unknown",
        "target_model": normalized_target_model,
        "user_gpu": str(device_name) if device_name is not None else "unknown",
        "draft_model": normalized_draft_model,
        "dataset": str(bench_name),
        "metric": str(objective_metric).lower(),
        "target_quantization": str(target_quantization or "8bit").lower(),
        "draft_quantization": str(draft_quantization or "8bit").lower(),
        "objective_selection_mode": str(objective_selection_mode or "blend").lower(),
        "reference_mode_key": str(reference_mode_key or "default"),
    }
    key_str = json.dumps(key_payload, sort_keys=True, ensure_ascii=True)
    key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()[:12]

    target_model_name = os.path.basename(normalized_target_model)
    draft_model_name = os.path.basename(normalized_draft_model)
    mode_name = _sanitize_key_component(str(objective_selection_mode or "blend").lower())
    bench_name_token = str(bench_name or "unknown").strip().lower().replace(" ", "-")
    filename = (
        f"ref_{_sanitize_key_component(server_name)}_"
        f"{_sanitize_key_component(target_model_name)}_"
        f"{_sanitize_key_component(device_name or 'unknown')}_"
        f"{_sanitize_key_component(draft_model_name)}_"
        f"tq-{_sanitize_key_component(str(target_quantization or '8bit').lower())}_"
        f"dq-{_sanitize_key_component(str(draft_quantization or '8bit').lower())}_"
        f"{bench_name_token}_"
        f"{_sanitize_key_component(objective_metric)}_"
        f"{mode_name}_"
        f"{key_hash}.json"
    )
    cache_path = os.path.join(reference_dir, filename)
    return reference_dir, cache_path


def load_reference_cache(
    base_model_path: str,
    draft_model_path: str,
    bench_name: str,
    objective_metric: str,
    server_name: str,
    device_name: str,
    target_quantization: str = "8bit",
    draft_quantization: str = "8bit",
    objective_selection_mode: str = None,
    reference_mode_key: str = None,
):
    _, cache_path = _reference_cache_paths(
        base_model_path=base_model_path,
        draft_model_path=draft_model_path,
        bench_name=bench_name,
        objective_metric=objective_metric,
        server_name=server_name,
        device_name=device_name,
        target_quantization=target_quantization,
        draft_quantization=draft_quantization,
        objective_selection_mode=objective_selection_mode,
        reference_mode_key=reference_mode_key,
    )
    if not os.path.exists(cache_path):
        # Legacy fallback: no quantization tokens.
        reference_dir = os.path.dirname(cache_path)
        normalized_target_model = _normalize_model_identifier(base_model_path)
        normalized_draft_model = _normalize_model_identifier(draft_model_path)
        target_model_name = os.path.basename(normalized_target_model)
        draft_model_name = os.path.basename(normalized_draft_model)
        mode_name = _sanitize_key_component(str(objective_selection_mode or "blend").lower())
        bench_name_token = str(bench_name or "unknown").strip().lower().replace(" ", "-")
        legacy_glob = (
            f"ref_{_sanitize_key_component(server_name)}_"
            f"{_sanitize_key_component(target_model_name)}_"
            f"{_sanitize_key_component(device_name or 'unknown')}_"
            f"{_sanitize_key_component(draft_model_name)}_"
            f"{bench_name_token}_"
            f"{_sanitize_key_component(objective_metric)}_"
            f"{mode_name}_*.json"
        )
        legacy_hits = sorted(glob.glob(os.path.join(reference_dir, legacy_glob)))
        if not legacy_hits:
            return None, cache_path
        cache_path = legacy_hits[-1]
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read reference cache ({cache_path}): {e}")
        return None, cache_path

    required = ["reference_tps", "reference_cost_per_token", "reference_objective_per_token"]
    if any(k not in data for k in required):
        print(f"[WARN] Invalid reference cache schema, ignoring: {cache_path}")
        return None, cache_path
    return data, cache_path


def save_reference_cache(
    cache_path: str,
    warmup_metrics: dict,
    objective_selection_mode: str = None,
):
    reference_cost_per_token = float(warmup_metrics.get("cost_per_token", 0.0))
    reference_objective_per_token = float(warmup_metrics.get("objective_per_token", 0.0))
    payload = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "warmup_cost_sensitivity": 0.5,
        "reference_tps": float(warmup_metrics.get("token_per_second", 0.0)),
        "reference_cost_per_token": reference_cost_per_token,
        "reference_objective_per_token": reference_objective_per_token,
        "reference_cost_per_1m_token": reference_cost_per_token * 1_000_000.0,
        "reference_objective_per_1m_token": reference_objective_per_token * 1_000_000.0,
        "reference_draft_objective_rate_per_sec": float(
            warmup_metrics.get("draft_objective_rate_per_sec", 0.0)
        ),
        "reference_target_objective_rate_per_sec": float(
            warmup_metrics.get("target_objective_rate_per_sec", 0.0)
        ),
        "feasible_metric_per_token": warmup_metrics.get("feasible_metric_per_token", None),
        "feasible_tps": warmup_metrics.get("feasible_tps", None),
        "reference_cs_anchor_curve": warmup_metrics.get("reference_cs_anchor_curve", None),
        "reference_tradeoff_curve_cs0_1": warmup_metrics.get("reference_tradeoff_curve_cs0_1", None),
        "reference_constraint_anchor_curve": warmup_metrics.get("reference_constraint_anchor_curve", None),
        "reference_tradeoff_curve_by_constraint": warmup_metrics.get("reference_tradeoff_curve_by_constraint", None),
        "reference_constraint_center_per_1m_token": warmup_metrics.get(
            "reference_constraint_center_per_1m_token", None
        ),
        "reference_point_repeat_count": warmup_metrics.get("reference_point_repeat_count", None),
        "reference_point_selection_rule": warmup_metrics.get("reference_point_selection_rule", None),
        "reference_fixed_accept_length_scale": warmup_metrics.get(
            "reference_fixed_accept_length_scale", None
        ),
        "reference_accept_length_last_trial": warmup_metrics.get(
            "reference_accept_length_last_trial", None
        ),
        "reference_monotonicity_summary": warmup_metrics.get(
            "reference_monotonicity_summary", None
        ),
        "reference_cause_summary": warmup_metrics.get(
            "reference_cause_summary", None
        ),
        "reference_debug_trace": warmup_metrics.get(
            "reference_debug_trace", None
        ),
        "objective_selection_mode": str(objective_selection_mode or "blend").lower(),
    }
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[WARN] Failed to write reference cache ({cache_path}): {e}")


def reset_gpu_monitor_after_reference(runner: "DraftRunner"):
    """Initializes GPU monitor internal buffers/counters after calculating reference."""
    gpu_monitor = getattr(runner, "gpu_monitor", None)
    if gpu_monitor is None:
        return
    try:
        gpu_monitor.stop_monitoring()
    except Exception:
        pass
    try:
        gpu_monitor.data = []
        gpu_monitor.monitor_call_count = 0
    except Exception:
        pass


def _build_reference_tradeoff_curve(anchor_rows: List[dict], step: float = 0.05) -> List[dict]:
    """
    Build cs 0~1 trade-off curve from cs anchors (e.g. 0/0.25/0.5/0.75/1).
    Piecewise-linear interpolation on predicted_tps and predicted_metric_per_1m_token.
    """
    if not isinstance(anchor_rows, list):
        return []
    by_cs = {}
    for row in anchor_rows:
        try:
            by_cs[round(float(row.get("cost_sensitivity")), 6)] = row
        except Exception:
            continue
    anchor_keys = sorted(by_cs.keys())
    if not anchor_keys or anchor_keys[0] > 0.0 or anchor_keys[-1] < 1.0:
        return []

    def interp(x0, y0, x1, y1, x):
        if x1 == x0:
            return float(y0)
        return float(y0) + (float(y1) - float(y0)) * ((float(x) - float(x0)) / (float(x1) - float(x0)))

    out = []
    step = max(1e-6, float(step))
    num_steps = int(round(1.0 / step))
    cs_values = [round(min(1.0, i * step), 6) for i in range(0, num_steps + 1)]
    if cs_values[-1] != 1.0:
        cs_values.append(1.0)
    for cs in cs_values:
        if cs in by_cs:
            a0 = by_cs[cs]
            a1 = by_cs[cs]
            x0 = cs
            x1 = cs
        else:
            right_idx = 0
            while right_idx < len(anchor_keys) and anchor_keys[right_idx] < cs:
                right_idx += 1
            if right_idx <= 0:
                x0 = anchor_keys[0]
                x1 = anchor_keys[1]
            elif right_idx >= len(anchor_keys):
                x0 = anchor_keys[-2]
                x1 = anchor_keys[-1]
            else:
                x0 = anchor_keys[right_idx - 1]
                x1 = anchor_keys[right_idx]
            a0 = by_cs[x0]
            a1 = by_cs[x1]
        tps = interp(x0, a0.get("predicted_tps", 0.0), x1, a1.get("predicted_tps", 0.0), cs)
        m1m = interp(
            x0, a0.get("predicted_metric_per_1m_token", 0.0),
            x1, a1.get("predicted_metric_per_1m_token", 0.0),
            cs,
        )
        out.append({
            "cost_sensitivity": float(cs),
            "predicted_tps": float(tps),
            "predicted_metric_per_1m_token": float(m1m),
        })
    return out


def _select_reference_anchor_for_cs(anchor_rows: List[dict], target_cs: float) -> Optional[dict]:
    """
    Select (or linearly interpolate) reference anchor by target cost_sensitivity.
    Returns a row-like dict containing predicted_tps/predicted_cost_per_token/
    predicted_objective_per_token and optional rate fields.
    """
    if not isinstance(anchor_rows, list) or not anchor_rows:
        return None

    parsed: List[Tuple[float, dict]] = []
    for row in anchor_rows:
        if not isinstance(row, dict):
            continue
        try:
            cs = float(row.get("cost_sensitivity"))
        except Exception:
            continue
        parsed.append((cs, row))
    if not parsed:
        return None
    parsed.sort(key=lambda x: x[0])

    cs_value = float(target_cs)

    def _to_float(v, default=0.0):
        try:
            return float(v)
        except Exception:
            return float(default)

    def _pick(row: dict, source_cs: float) -> dict:
        out = dict(row)
        out["_selected_ref_cs"] = float(source_cs)
        out["_selected_ref_cs_left"] = float(source_cs)
        out["_selected_ref_cs_right"] = float(source_cs)
        out["_selected_ref_interpolated"] = False
        return out

    if cs_value <= parsed[0][0]:
        return _pick(parsed[0][1], parsed[0][0])
    if cs_value >= parsed[-1][0]:
        return _pick(parsed[-1][1], parsed[-1][0])

    for i in range(1, len(parsed)):
        cs0, r0 = parsed[i - 1]
        cs1, r1 = parsed[i]
        if abs(cs_value - cs0) <= 1e-12:
            return _pick(r0, cs0)
        if abs(cs_value - cs1) <= 1e-12:
            return _pick(r1, cs1)
        if cs0 < cs_value < cs1:
            ratio = (cs_value - cs0) / max(1e-12, (cs1 - cs0))

            def _interp(key: str, default: float = 0.0) -> float:
                v0 = _to_float(r0.get(key, default), default)
                v1 = _to_float(r1.get(key, default), default)
                return v0 + (v1 - v0) * ratio

            out = {
                "cost_sensitivity": float(cs_value),
                "predicted_tps": _interp("predicted_tps", 0.0),
                "predicted_cost_per_token": _interp("predicted_cost_per_token", 0.0),
                "predicted_objective_per_token": _interp("predicted_objective_per_token", 0.0),
                "predicted_metric_per_1m_token": _interp("predicted_metric_per_1m_token", 0.0),
                "predicted_objective_per_1m_token": _interp("predicted_objective_per_1m_token", 0.0),
                "draft_objective_rate_per_sec": _interp("draft_objective_rate_per_sec", 0.0),
                "target_objective_rate_per_sec": _interp("target_objective_rate_per_sec", 0.0),
                "_selected_ref_cs": float(cs_value),
                "_selected_ref_cs_left": float(cs0),
                "_selected_ref_cs_right": float(cs1),
                "_selected_ref_interpolated": True,
            }
            return out
    return None


def _analyze_reference_tradeoff_curve(curve_rows: List[dict]) -> dict:
    if not isinstance(curve_rows, list) or not curve_rows:
        return {
            "point_count": 0,
            "metric_monotonic_nonincreasing": None,
            "tps_monotonic_nonincreasing": None,
            "metric_violation_segments": [],
            "tps_violation_segments": [],
        }
    rows = []
    for row in curve_rows:
        try:
            rows.append({
                "cost_sensitivity": float(row.get("cost_sensitivity", 0.0)),
                "predicted_tps": float(row.get("predicted_tps", 0.0)),
                "predicted_metric_per_1m_token": float(row.get("predicted_metric_per_1m_token", 0.0)),
            })
        except Exception:
            continue
    rows.sort(key=lambda x: x["cost_sensitivity"])
    metric_bad = []
    tps_bad = []
    for i in range(1, len(rows)):
        prev_row = rows[i - 1]
        cur_row = rows[i]
        dm = float(cur_row["predicted_metric_per_1m_token"] - prev_row["predicted_metric_per_1m_token"])
        dt = float(cur_row["predicted_tps"] - prev_row["predicted_tps"])
        if dm > 1e-12:
            metric_bad.append({
                "from_cs": float(prev_row["cost_sensitivity"]),
                "to_cs": float(cur_row["cost_sensitivity"]),
                "delta_metric_per_1m": dm,
            })
        if dt > 1e-12:
            tps_bad.append({
                "from_cs": float(prev_row["cost_sensitivity"]),
                "to_cs": float(cur_row["cost_sensitivity"]),
                "delta_tps": dt,
            })
    return {
        "point_count": int(len(rows)),
        "metric_monotonic_nonincreasing": bool(len(metric_bad) == 0),
        "tps_monotonic_nonincreasing": bool(len(tps_bad) == 0),
        "metric_violation_segments": metric_bad,
        "tps_violation_segments": tps_bad,
    }


def _build_reference_cause_summary(anchor_rows: List[dict], curve_rows: List[dict]) -> dict:
    by_cs_summary = {}
    if isinstance(anchor_rows, list):
        for row in anchor_rows:
            try:
                cs = f"{float(row.get('cost_sensitivity')):.6g}"
            except Exception:
                continue
            dbg = row.get("reference_debug_trace") if isinstance(row, dict) else None
            if isinstance(dbg, dict):
                by_cs_summary[cs] = dbg.get("by_cs_summary", {}).get(str(cs), None)
    curve = []
    if isinstance(curve_rows, list):
        for row in curve_rows:
            try:
                curve.append({
                    "cs": float(row.get("cost_sensitivity", 0.0)),
                    "metric": float(row.get("predicted_metric_per_1m_token", 0.0)),
                    "tps": float(row.get("predicted_tps", 0.0)),
                })
            except Exception:
                continue
    curve.sort(key=lambda x: x["cs"])
    transitions = []
    for i in range(1, len(curve)):
        prev_row = curve[i - 1]
        cur_row = curve[i]
        cs0 = f"{float(prev_row['cs']):.6g}"
        cs1 = f"{float(cur_row['cs']):.6g}"
        s0 = by_cs_summary.get(cs0)
        s1 = by_cs_summary.get(cs1)
        transitions.append({
            "from_cs": float(prev_row["cs"]),
            "to_cs": float(cur_row["cs"]),
            "delta_metric_per_1m": float(cur_row["metric"] - prev_row["metric"]),
            "delta_tps": float(cur_row["tps"] - prev_row["tps"]),
            "from_tree": (
                {
                    "avg_tree_depth": s0.get("avg_tree_depth"),
                    "avg_final_nnodes": s0.get("avg_final_nnodes"),
                    "avg_width_over_depths": s0.get("avg_width_over_depths"),
                    "dominant_cost_component": s0.get("dominant_cost_component"),
                } if isinstance(s0, dict) else None
            ),
            "to_tree": (
                {
                    "avg_tree_depth": s1.get("avg_tree_depth"),
                    "avg_final_nnodes": s1.get("avg_final_nnodes"),
                    "avg_width_over_depths": s1.get("avg_width_over_depths"),
                    "dominant_cost_component": s1.get("dominant_cost_component"),
                } if isinstance(s1, dict) else None
            ),
        })
    return {
        "anchor_cs_available": sorted(by_cs_summary.keys()),
        "transition_analysis": transitions,
    }


def _build_reference_tradeoff_curve_by_constraint(anchor_rows: List[dict], constraint_target: str = "metric") -> List[dict]:
    """
    Build constraint-mode trade-off points sorted by active constraint selector.
    Used when objective_selection_mode=constraint.
    """
    if not isinstance(anchor_rows, list):
        return []
    constraint_target = str(constraint_target or "metric").lower()
    rows = []
    for row in anchor_rows:
        try:
            tps = float(row.get("predicted_tps", 0.0))
            m1m = float(
                row.get(
                    "predicted_metric_per_1m_token",
                    row.get("predicted_objective_per_1m_token", 0.0),
                )
            )
            out_row = {
                "predicted_tps": tps,
                "predicted_metric_per_1m_token": m1m,
            }
            if constraint_target == "tps":
                out_row["min_tps_constraint"] = float(row.get("min_tps_constraint", tps))
            else:
                out_row["metric_constraint_per_1m_token"] = float(row.get("metric_constraint_per_1m_token"))
            rows.append(out_row)
        except Exception:
            continue
    sort_key = "min_tps_constraint" if constraint_target == "tps" else "metric_constraint_per_1m_token"
    rows.sort(key=lambda x: x[sort_key])
    return rows


def _parse_reference_constraint_multipliers(value: str) -> List[float]:
    """
    Parse comma-separated multipliers used for reference constraint sweep.
    Example: "0.8,1.0,1.2"
    """
    if value is None:
        return [0.8, 1.0, 1.2]
    out = []
    for tok in str(value).split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = float(tok)
        except Exception:
            continue
        if np.isfinite(v) and v > 0:
            out.append(v)
    if not out:
        return [0.8, 1.0, 1.2]
    # Remove near-duplicate values while preserving sort order.
    out = sorted(out)
    dedup = []
    for v in out:
        if not dedup or abs(v - dedup[-1]) > 1e-9:
            dedup.append(v)
    return dedup


def load_server_only_baseline_metric(
    baseline_json: str,
    base_model_path: str,
    draft_model_path: str,
):
    """Server-only baseline tps/cost_per_token (cost_per_token derived from target_per_sec_cost at call site)."""
    if not baseline_json or not os.path.exists(baseline_json):
        print(f"[INFO] server-only disabled: baseline file not found ({baseline_json}).")
        return None
    try:
        with open(baseline_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        key = f"{base_model_path}|{draft_model_path}"
        pair = data.get("model_pairs", {}).get(key)
        if not isinstance(pair, dict):
            print(f"[INFO] server-only disabled: no baseline for model pair '{key}'.")
            return None
        tps = pair.get("token_per_second", None)
        if tps is None:
            print(f"[INFO] server-only disabled: token_per_second missing for '{key}'.")
            return None
        tps = float(tps)
        if abs(tps - 1.0) < 1e-9:
            print(
                "[WARN] server-only baseline tps is 1.0 (likely placeholder). "
                "Update data/profile/server_only_sd_baseline.json with measured value for reliable prediction."
            )
        if tps <= 0:
            print(f"[INFO] server-only disabled: invalid token_per_second={tps} for '{key}'.")
            return None
        return tps
    except Exception as e:
        print(f"[INFO] server-only disabled: failed to load baseline ({e}).")
        return None


def run_server_only_probe(
    sock,
    questions: List[dict],
    tokenizer: AutoTokenizer,
    base_model_path: str,
    bench_name: str,
    nodes: int,
    max_depth: int,
    per_token_probability_bound: float,
    per_path_probability_bound: float,
    min_width: int,
    fixed_width: bool,
    fixed_width_value: int,
    fixed_nnodes: bool,
    fixed_depth: bool,
    proactive_drafting: bool,
    proactive_threshold: float,
    adaptive_proactive_threshold: bool,
    cost_sensitivity: float,
    draft_per_sec_cost: float,
    target_per_sec_cost: float,
    reference_tps: float,
    reference_objective_per_token: float,
    objective_metric: str,
    no_draft_cost: bool,
    bill_draft_as_target_gpu: bool = False,
    server_draft_profile_auto: bool = True,
    server_draft_profile_force_refresh: bool = False,
    server_draft_profile_model_calls_per_count: int = 100,
    server_draft_profile_width_list: str = "10,20,30,40,50,60,70,80,90,100,110,120,130,140,150",
    question_file: str = "",
    server_name: str = "rtxproa6000",
) -> dict:
    """Runs one short server-only probe. (For verification of prediction range)"""
    if not questions:
        return {"token_per_second": 0.0, "metric_per_token": 0.0, "new_tokens": 0}

    warm_q = questions[0]
    conv = _build_conversation_template_for_model(base_model_path)
    warm_turns = _extract_question_turns(warm_q, bench_name)
    conv.append_message(conv.roles[0], warm_turns[0])
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt() + " "
    input_ids = tokenizer([prompt]).input_ids

    prev_sock_timeout = sock.gettimeout()
    init_timeout_s = _estimate_server_only_init_timeout_seconds(
        server_draft_profile_auto=bool(server_draft_profile_auto),
        server_draft_profile_force_refresh=bool(server_draft_profile_force_refresh),
        server_draft_profile_model_calls_per_count=int(max(1, int(server_draft_profile_model_calls_per_count))),
        server_draft_profile_width_list=str(server_draft_profile_width_list),
    )
    sock.settimeout(init_timeout_s)
    send_json_with_size(sock, {
        "type": "server_only_init",
        "input_ids": input_ids[0],
        "nodes": int(nodes),
        "max_depth": int(max_depth),
        "per_token_probability_bound": float(per_token_probability_bound),
        "per_path_probability_bound": float(per_path_probability_bound),
        "min_width": int(min_width),
        "fixed_width": bool(fixed_width),
        "fixed_width_value": fixed_width_value,
        "fixed_nnodes": bool(fixed_nnodes),
        "fixed_depth": bool(fixed_depth),
        "proactive_drafting": bool(proactive_drafting),
        "proactive_threshold": float(proactive_threshold),
        "adaptive_proactive_threshold": bool(adaptive_proactive_threshold),
        "cost_sensitivity": float(cost_sensitivity),
        "draft_per_sec_cost": float(draft_per_sec_cost),
        "target_per_sec_cost": float(target_per_sec_cost),
        "bill_draft_as_target_gpu": bool(bill_draft_as_target_gpu),
        "server_draft_profile_auto": bool(server_draft_profile_auto),
        "server_draft_profile_force_refresh": bool(server_draft_profile_force_refresh),
        "server_draft_profile_model_calls_per_count": int(max(1, int(server_draft_profile_model_calls_per_count))),
        "server_draft_profile_width_list": str(server_draft_profile_width_list),
        "server_name": str(server_name),
        "bench_name": str(bench_name),
        "reference_tps": float(reference_tps),
        "reference_objective_per_token": float(reference_objective_per_token),
        "objective_metric": str(objective_metric),
        "no_draft_cost": bool(no_draft_cost),
    })
    try:
        reply, _ = recv_json_with_size(sock)
    finally:
        sock.settimeout(prev_sock_timeout)
    if reply.get("type") != "server_only_ok":
        return {"token_per_second": 0.0, "metric_per_token": 0.0, "new_tokens": 0}

    total_new_tokens = 0
    total_target_energy_kwh = 0.0
    probe_start = time.time()
    while True:
        reply, _ = recv_json_with_size(sock)
        if reply.get("type") == "server_only_done":
            break
        if reply.get("type") != "verify_result":
            break
        accepted_tokens = reply.get("accepted_tokens", []) or []
        # server-only accepted
        total_new_tokens += int(len(accepted_tokens))
        if str(objective_metric).lower() == "target_energy":
            target_verification_sec = max(
                0.0, float(reply.get("target_verification_time_ms", 0.0) or 0.0) / 1000.0
            )
            target_energy_rate_per_sec = reply.get("target_energy_rate_per_sec", None)
            try:
                target_energy_rate_per_sec = float(target_energy_rate_per_sec)
            except Exception:
                target_energy_rate_per_sec = None
            if target_verification_sec > 0 and target_energy_rate_per_sec is not None and target_energy_rate_per_sec > 0:
                total_target_energy_kwh += target_energy_rate_per_sec * target_verification_sec

    probe_elapsed = max(1e-9, float(time.time() - probe_start))
    probe_tps = float(total_new_tokens) / probe_elapsed if total_new_tokens > 0 else 0.0
    if str(objective_metric).lower() in {"cost", "total_cost", "api_cost"}:
        metric_per_token = (float(target_per_sec_cost) / probe_tps) if probe_tps > 0 else float("inf")
    else:
        metric_per_token = (
            float(total_target_energy_kwh) / float(total_new_tokens)
            if total_new_tokens > 0
            else 0.0
        )
    return {
        "token_per_second": float(probe_tps),
        "metric_per_token": float(metric_per_token),
        "new_tokens": int(total_new_tokens),
        "elapsed_sec": float(probe_elapsed),
    }


def _get_profile_paths(
    draft_model_path: str,
    base_model_path: str,
    bench_name: str,
    device_name: str,
    server_name: str,
    target_quantization: str,
    draft_quantization: str,
):
    # NOTE:
    # Draft profile path is objective-agnostic by design.
    # total_cost and draft_energy both consume the same draft energy fields
    # (power/energy), so they intentionally share one draft profile artifact.
    script_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(script_dir)
    profile_dir = os.path.join(_resolve_data_root(parent_dir), "profile")
    draft_model_name = draft_model_path.split("/")[-1] if "/" in draft_model_path else draft_model_path
    base_model_name = base_model_path.split("/")[-1] if "/" in base_model_path else base_model_path
    tq_name = _sanitize_key_component(str(target_quantization or "8bit").lower())
    dq_name = _sanitize_key_component(str(draft_quantization or "8bit").lower())
    draft_profile_file = os.path.join(
        profile_dir,
        f"profile_draft_{_sanitize_key_component(device_name)}_{_sanitize_key_component(draft_model_name)}_dq-{dq_name}.json",
    )
    target_profile_file = os.path.join(
        profile_dir,
        f"profile_target_{_sanitize_key_component(server_name)}_{_sanitize_key_component(base_model_name)}_tq-{tq_name}.json",
    )
    return draft_profile_file, target_profile_file


def _resolve_profile_variant_path(preferred_path: str, profile_tag: str) -> str:
    """
    Resolve profile path robustly when quantization fallback produced a different suffix.
    profile_tag: 'dq' for draft profiles, 'tq' for target profiles.
    """
    path = os.path.expanduser(str(preferred_path or ""))
    if (not path) or os.path.exists(path):
        return path
    profile_tag = str(profile_tag or "").strip().lower()
    if profile_tag not in {"dq", "tq"}:
        return path
    base = os.path.basename(path)
    marker = f"_{profile_tag}-"
    if marker not in base:
        return path
    prefix = base.split(marker, 1)[0] + marker
    parent = os.path.dirname(path) or "."
    pattern = os.path.join(parent, f"{prefix}*.json")
    variants = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if variants:
        return variants[0]
    return path


def _parse_int_list_csv(raw: Optional[str], default_values: List[int]) -> List[int]:
    if raw is None:
        return list(default_values)
    vals = []
    for tok in str(raw).split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            vals.append(int(tok))
        except Exception:
            continue
    return vals or list(default_values)


def _safe_json_dump_atomic(path: str, payload: Dict[str, Any]) -> bool:
    try:
        if not path:
            return False
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp_profile_", suffix=".json", dir=(parent or None))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
        return True
    except Exception:
        return False


def _sanitize_online_lr(value: float) -> float:
    try:
        lr = float(value)
    except Exception:
        return float(ONLINE_PROFILE_LR_DEFAULT)
    if not np.isfinite(lr):
        return float(ONLINE_PROFILE_LR_DEFAULT)
    return max(1e-6, min(1.0, lr))


def _ema_update_scalar(old_value: Optional[float], observed_value: Optional[float], lr: float) -> Optional[float]:
    try:
        obs = float(observed_value)
    except Exception:
        return old_value
    if (not np.isfinite(obs)) or obs < 0:
        return old_value
    try:
        old = float(old_value)
    except Exception:
        old = obs
    if (not np.isfinite(old)) or old < 0:
        old = obs
    return float(old + lr * (obs - old))


def _summarize_draft_profile_missing_stats(profile_file: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Aggregate per-run missing GPU metric statistics by width from the draft profile JSON.
    - Valid/missing ratios are computed by sample count (number of measurement events)
    - Older profiles without missing-stat fields return available=False
    """
    if not profile_file:
        return None
    path = os.path.expanduser(profile_file)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None

    width_rows: List[Tuple[int, Dict[str, Any]]] = []
    for k, v in raw.items():
        try:
            w = int(k)
        except Exception:
            continue
        if isinstance(v, dict):
            width_rows.append((w, v))
    width_rows.sort(key=lambda x: x[0])
    if not width_rows:
        return None

    totals = {
        "monitor_sample_count": 0,
        "monitor_error_count": 0,
        "power_valid_count": 0,
        "power_missing_count": 0,
        "util_valid_count": 0,
        "util_missing_count": 0,
        "graphics_clock_valid_count": 0,
        "graphics_clock_missing_count": 0,
        "memory_clock_valid_count": 0,
        "memory_clock_missing_count": 0,
    }
    fields_present = False

    for _, row in width_rows:
        if any(
            key in row for key in (
                "gpu_monitor_sample_count",
                "gpu_power_sample_valid_count",
                "gpu_power_sample_missing_count",
            )
        ):
            fields_present = True
        totals["monitor_sample_count"] += int(row.get("gpu_monitor_sample_count", 0) or 0)
        totals["monitor_error_count"] += int(row.get("gpu_monitor_error_count", 0) or 0)
        totals["power_valid_count"] += int(row.get("gpu_power_sample_valid_count", 0) or 0)
        totals["power_missing_count"] += int(row.get("gpu_power_sample_missing_count", 0) or 0)
        totals["util_valid_count"] += int(row.get("gpu_util_sample_valid_count", 0) or 0)
        totals["util_missing_count"] += int(row.get("gpu_util_sample_missing_count", 0) or 0)
        totals["graphics_clock_valid_count"] += int(row.get("gpu_graphics_clock_sample_valid_count", 0) or 0)
        totals["graphics_clock_missing_count"] += int(row.get("gpu_graphics_clock_sample_missing_count", 0) or 0)
        totals["memory_clock_valid_count"] += int(row.get("gpu_memory_clock_sample_valid_count", 0) or 0)
        totals["memory_clock_missing_count"] += int(row.get("gpu_memory_clock_sample_missing_count", 0) or 0)

    def _pct(missing: int, total: int) -> float:
        return float(missing * 100.0 / total) if total > 0 else 0.0

    denom = int(totals["monitor_sample_count"])
    summary = {
        "available": bool(fields_present),
        "width_count": int(len(width_rows)),
        "monitor_sample_count": denom,
        "monitor_error_count": int(totals["monitor_error_count"]),
        "power_missing_count": int(totals["power_missing_count"]),
        "power_missing_percent": _pct(int(totals["power_missing_count"]), denom),
        "util_missing_count": int(totals["util_missing_count"]),
        "util_missing_percent": _pct(int(totals["util_missing_count"]), denom),
        "graphics_clock_missing_count": int(totals["graphics_clock_missing_count"]),
        "graphics_clock_missing_percent": _pct(int(totals["graphics_clock_missing_count"]), denom),
        "memory_clock_missing_count": int(totals["memory_clock_missing_count"]),
        "memory_clock_missing_percent": _pct(int(totals["memory_clock_missing_count"]), denom),
        "power_valid_count": int(totals["power_valid_count"]),
        "util_valid_count": int(totals["util_valid_count"]),
        "graphics_clock_valid_count": int(totals["graphics_clock_valid_count"]),
        "memory_clock_valid_count": int(totals["memory_clock_valid_count"]),
    }
    if not fields_present:
        summary["note"] = "Missing-stat fields are not present in this draft profile file (legacy profile)."
    return summary


def _update_online_draft_profile(
    profile_data: Optional[Dict[str, Any]],
    observed_width_ms: Dict[int, List[float]],
    lr: float,
) -> Tuple[Optional[Dict[str, Any]], int]:
    if not isinstance(profile_data, dict) or not profile_data:
        return profile_data, 0
    changed_rows = 0
    for width, ms_list in (observed_width_ms or {}).items():
        if not ms_list:
            continue
        vals = []
        for v in ms_list:
            try:
                x = float(v)
                if np.isfinite(x) and x >= 0:
                    vals.append(x)
            except Exception:
                continue
        if not vals:
            continue
        key = str(int(width))
        row = profile_data.get(key)
        if not isinstance(row, dict):
            row = {
                "model_call_count": 0,
                "model_call_total_time_ms": 0.0,
                "model_call_avg_time_ms": float(sum(vals) / len(vals)),
                "model_call_min_time_ms": float(min(vals)),
                "model_call_max_time_ms": float(max(vals)),
            }
            profile_data[key] = row
        observed_avg = float(sum(vals) / len(vals))
        old_avg = row.get("model_call_avg_time_ms", None)
        new_avg = _ema_update_scalar(old_avg, observed_avg, lr)
        if new_avg is None:
            continue
        old_min = row.get("model_call_min_time_ms", None)
        old_max = row.get("model_call_max_time_ms", None)
        try:
            old_count = int(row.get("model_call_count", 0))
        except Exception:
            old_count = 0
        added_count = len(vals)
        new_count = max(0, old_count + added_count)
        row["model_call_count"] = int(new_count)
        row["model_call_avg_time_ms"] = float(new_avg)
        row["model_call_total_time_ms"] = float(new_avg) * float(new_count)
        observed_min = float(min(vals))
        observed_max = float(max(vals))
        if old_min is None:
            row["model_call_min_time_ms"] = observed_min
        else:
            try:
                row["model_call_min_time_ms"] = float(min(float(old_min), observed_min))
            except Exception:
                row["model_call_min_time_ms"] = observed_min
        if old_max is None:
            row["model_call_max_time_ms"] = observed_max
        else:
            try:
                row["model_call_max_time_ms"] = float(max(float(old_max), observed_max))
            except Exception:
                row["model_call_max_time_ms"] = observed_max
        row["online_update_count"] = int(max(0, int(row.get("online_update_count", 0) or 0) + 1))
        changed_rows += 1
    return profile_data, changed_rows


def _update_online_target_profile(
    target_profile_data: Optional[Dict[str, Any]],
    observed_nnodes_ms: Dict[int, List[float]],
    lr: float,
) -> Tuple[Optional[Dict[str, Any]], int]:
    if not isinstance(target_profile_data, dict) or not target_profile_data:
        return target_profile_data, 0
    changed_rows = 0
    for nnodes, ms_list in (observed_nnodes_ms or {}).items():
        if not ms_list:
            continue
        vals = []
        for v in ms_list:
            try:
                x = float(v)
                if np.isfinite(x) and x >= 0:
                    vals.append(x)
            except Exception:
                continue
        if not vals:
            continue
        key = f"nnodes_{int(nnodes)}"
        row = target_profile_data.get(key)
        if not isinstance(row, dict):
            row = {
                "max_nnodes": int(nnodes),
                "count": 0,
                "total_time_ms": 0.0,
                "avg_time_ms": float(sum(vals) / len(vals)),
                "min_time_ms": float(min(vals)),
                "max_time_ms": float(max(vals)),
            }
            target_profile_data[key] = row
        observed_avg = float(sum(vals) / len(vals))
        old_avg = row.get("avg_time_ms", None)
        new_avg = _ema_update_scalar(old_avg, observed_avg, lr)
        if new_avg is None:
            continue
        try:
            old_count = int(row.get("count", 0))
        except Exception:
            old_count = 0
        added_count = len(vals)
        new_count = max(0, old_count + added_count)
        row["max_nnodes"] = int(nnodes)
        row["count"] = int(new_count)
        row["avg_time_ms"] = float(new_avg)
        row["total_time_ms"] = float(new_avg) * float(new_count)
        obs_min = float(min(vals))
        obs_max = float(max(vals))
        try:
            row["min_time_ms"] = float(min(float(row.get("min_time_ms", obs_min)), obs_min))
        except Exception:
            row["min_time_ms"] = obs_min
        try:
            row["max_time_ms"] = float(max(float(row.get("max_time_ms", obs_max)), obs_max))
        except Exception:
            row["max_time_ms"] = obs_max
        # keep compatible aliases in sync
        row["avg_verification_time_ms"] = float(new_avg)
        row["total_verification_time_ms"] = float(new_avg) * float(new_count)
        row["online_update_count"] = int(max(0, int(row.get("online_update_count", 0) or 0) + 1))
        changed_rows += 1
    return target_profile_data, changed_rows


def _apply_online_profile_updates_and_flush(
    runner: DraftRunner,
    *,
    observed_width_ms: Dict[int, List[float]],
    observed_nnodes_ms: Dict[int, List[float]],
    debug: bool = False,
) -> Dict[str, Any]:
    started = time.perf_counter()
    result = {
        "enabled": bool(getattr(runner, "online_profile_update_enabled", True)),
        "draft_rows_updated": 0,
        "target_rows_updated": 0,
        "draft_saved": False,
        "target_saved": False,
        "overhead_sec": 0.0,
    }
    if not result["enabled"]:
        result["overhead_sec"] = max(0.0, float(time.perf_counter() - started))
        return result
    lr = _sanitize_online_lr(getattr(runner, "online_profile_lr", ONLINE_PROFILE_LR_DEFAULT))
    try:
        updated_profile, draft_rows = _update_online_draft_profile(
            profile_data=copy.deepcopy(runner.profile_data) if isinstance(runner.profile_data, dict) else None,
            observed_width_ms=observed_width_ms,
            lr=lr,
        )
        if draft_rows > 0 and isinstance(updated_profile, dict):
            runner.profile_data = updated_profile
            result["draft_rows_updated"] = int(draft_rows)
            if runner.draft_profile_file:
                result["draft_saved"] = bool(_safe_json_dump_atomic(runner.draft_profile_file, updated_profile))
    except Exception as e:
        if debug:
            print(f"[online-profile][WARN] draft update failed: {e}")
    try:
        updated_target_profile, target_rows = _update_online_target_profile(
            target_profile_data=copy.deepcopy(runner.target_profile_data) if isinstance(runner.target_profile_data, dict) else None,
            observed_nnodes_ms=observed_nnodes_ms,
            lr=lr,
        )
        if target_rows > 0 and isinstance(updated_target_profile, dict):
            runner.target_profile_data = updated_target_profile
            result["target_rows_updated"] = int(target_rows)
            if runner.target_profile_file:
                result["target_saved"] = bool(_safe_json_dump_atomic(runner.target_profile_file, updated_target_profile))
    except Exception as e:
        if debug:
            print(f"[online-profile][WARN] target update failed: {e}")
    result["overhead_sec"] = max(0.0, float(time.perf_counter() - started))
    return result


def _request_target_profile_generation(
    sock: socket.socket,
    *,
    runner: "DraftRunner",
    tokenizer: AutoTokenizer,
    base_model_path: str,
    draft_model_path: str,
    bench_name: str,
    question_file: str,
    draft_quantization: str,
    device_map: str,
    width_list: List[int],
    depth_list: List[int],
    node_list: List[int],
    runs_per_combination: int,
    force_refresh: bool,
    poll_interval_sec: float,
    timeout_sec: float,
    debug: bool,
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    try:
        questions = _load_benchmark_questions(bench_name, question_file)
        if not questions:
            raise RuntimeError("target profiling requires at least one benchmark question")
        profile_q = questions[0]
        conv = _build_conversation_template_for_model(draft_model_path)
        turns = _extract_question_turns(profile_q, bench_name)
        conv.append_message(conv.roles[0], turns[0])
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt() + " "
        input_ids = tokenizer([prompt]).input_ids
        if not input_ids or not input_ids[0]:
            raise RuntimeError("failed to tokenize profiling prompt for target profiling")
        base_input_ids = [int(x) for x in input_ids[0]]
        send_json_with_size(sock, {"type": "init", "input_ids": base_input_ids})
        init_reply, _ = recv_json_with_size(sock)
        if str(init_reply.get("type", "")) != "init_ok":
            raise RuntimeError(f"failed to get init token for target profiling: {init_reply}")
        next_token = int(init_reply.get("next_token", -1))
        if next_token < 0:
            raise RuntimeError(f"invalid next_token from target init: {init_reply}")
        node_values = [int(x) for x in node_list if int(x) > 0]
        if not node_values:
            raise RuntimeError("target profiling requires non-empty node_list")
        fixed_width = int(max(width_list or [50]))
        fixed_depth = int(max(depth_list or [10]))
        max_nodes = int(max(node_values))
        runner.reset_kv()
        input_ids_t = torch.as_tensor([base_input_ids]).to(runner.draft_model.lm_head.weight.device)
        draft_profile_start = time.monotonic()
        (
            draft_ids,
            draft_pos,
            tree_mask,
            parent,
            tree_depth,
            _final_nnodes,
            _depth_widths,
            _node_meta,
        ) = build_tree_with_next_token(
            runner=runner,
            input_ids=input_ids_t,
            nodes=max_nodes,
            max_depth=fixed_depth,
            next_token_id=next_token,
            tokenizer=tokenizer,
            debug=bool(debug),
            print_tree=False,
            per_token_probability_bound=0.0,
            per_path_probability_bound=0.0,
            fixed_width=True,
            fixed_width_value=fixed_width,
            fixed_nnodes=True,
            fixed_depth=True,
        )
        base_tree_build_time_sec = max(0.0, float(time.monotonic() - draft_profile_start))
        parent_list = parent.tolist() if isinstance(parent, torch.Tensor) else parent
        base_tree_payload = {
            "width": int(fixed_width),
            "depth": int(fixed_depth),
            "max_nnodes": int(max_nodes),
            "draft_input_ids": draft_ids[0].tolist(),
            "draft_position_ids": draft_pos.tolist(),
            "tree_attention_mask": tree_mask.tolist(),
            "parent": [int(x) for x in parent_list],
            "tree_depth": int(tree_depth),
            "draft_profile_time_sec": float(base_tree_build_time_sec),
            "next_token": int(next_token),
        }
        start_payload = {
            "type": "profile_target_start",
            "base_model_path": base_model_path,
            "draft_model_path": draft_model_path,
            "bench_name": bench_name,
            # "question_file": question_file,
            # Do not send client-local absolute paths to target.
            # Target resolves question file from its own repo by bench_name.
            "draft_quantization": str(draft_quantization or "none"),
            "draft_device_map": str(device_map or "cuda:0"),
            "width_list": [int(x) for x in width_list],
            "depth_list": [int(x) for x in depth_list],
            "node_list": [int(x) for x in node_list],
            "runs_per_combination": int(max(1, runs_per_combination)),
            "profile_deterministic": True,
            "profile_warmup_runs": int(max(0, PROFILE_WARMUP_RUNS)),
            "profile_burnin_runs": int(max(0, PROFILE_BURNIN_RUNS)),
            "force_refresh": bool(force_refresh),
            "base_input_ids": base_input_ids,
            "base_tree": base_tree_payload,
        }
        send_json_with_size(sock, start_payload)
        start_reply, _ = recv_json_with_size(sock)
        if start_reply.get("type") not in {"profile_target_started"}:
            if debug:
                print(f"[TargetProfile][WARN] start failed: {start_reply}")
            return False, None, None
        deadline = time.time() + max(10.0, float(timeout_sec))
        while time.time() < deadline:
            time.sleep(max(0.1, float(poll_interval_sec)))
            send_json_with_size(sock, {"type": "profile_target_status"})
            status_reply, _ = recv_json_with_size(sock)
            if status_reply.get("type") != "profile_target_status_ok":
                if debug:
                    print(f"[TargetProfile][WARN] unexpected status reply: {status_reply}")
                continue
            status = status_reply.get("status", {})
            state = str(status.get("status", "idle"))
            if state == "done":
                if debug:
                    print(f"[TargetProfile] done: {status.get('output_file')}")
                send_json_with_size(sock, {"type": "profile_target_get_result"})
                result_reply, _ = recv_json_with_size(sock)
                if result_reply.get("type") != "profile_target_result_ok":
                    if debug:
                        print(f"[TargetProfile][WARN] unexpected result reply: {result_reply}")
                    return False, None, None
                timing_summary = result_reply.get("timing_summary")
                if not isinstance(timing_summary, dict):
                    timing_summary = None
                has_explicit_success = "success" in result_reply
                if has_explicit_success and not bool(result_reply.get("success", False)):
                    if debug:
                        print(f"[TargetProfile][ERROR] result payload missing: {result_reply.get('error')}")
                    return False, None, timing_summary
                profile_payload = result_reply.get("profile_data")
                if isinstance(profile_payload, dict) and profile_payload:
                    return True, profile_payload, timing_summary
                # Backward compatibility: older target may not send success/profile_data fields.
                if (not has_explicit_success) and str(status.get("status", "")) == "done":
                    if debug:
                        print("[TargetProfile][WARN] legacy target response detected (no success/profile_data)")
                    return True, None, timing_summary
                if debug:
                    print("[TargetProfile][ERROR] result payload is empty")
                return False, None, timing_summary
            if state == "error":
                if debug:
                    print(f"[TargetProfile][ERROR] profiling failed: {status.get('error')}")
                return False, None, None
            if debug and state in {"running", "starting"}:
                done_cnt = status.get("completed_combinations", 0)
                total_cnt = status.get("total_combinations", 0)
                current_combo = status.get("current_combination", None)
                print(f"[TargetProfile] {state} {done_cnt}/{total_cnt}, current={current_combo}")
        if debug:
            print("[TargetProfile][WARN] timeout while waiting profile_target_status")
        return False, None, None
    except Exception as e:
        if debug:
            print(f"[TargetProfile][ERROR] exception while requesting profile: {e}")
        return False, None, None


def load_profile_data(
    runner: DraftRunner,
    tokenizer: AutoTokenizer,
    max_depth: int,
    draft_model_path: str,
    base_model_path: str,
    bench_name: str,
    device_name: str,
    server_name: str,
    target_quantization: str,
    draft_quantization: str,
    question_file: str,
    fixed_depth: bool,
    debug: bool,
    target_sock: Optional[socket.socket] = None,
    auto_target_profile: bool = False,
    draft_profile_force_refresh: bool = False,
    draft_profile_model_calls_per_count: int = 100,
    draft_profile_width_list: str = "10,20,30,40,50,60,70,80,90,100,110,120,130,140,150",
    target_profile_force_refresh: bool = False,
    target_profile_model_calls_per_count: int = 20,
    target_profile_width_list: Optional[str] = None,
    target_profile_depth_list: Optional[str] = None,
    target_profile_node_list: Optional[str] = None,
    target_profile_poll_interval_sec: float = 2.0,
    target_profile_timeout_sec: float = 7200.0,
    draft_device_map: str = "cuda:0",
    online_profile_update: bool = True,
    online_profile_lr: float = ONLINE_PROFILE_LR_DEFAULT,
    bill_draft_as_target_gpu: bool = False,
    server_draft_profile_auto: bool = True,
    server_draft_profile_force_refresh: bool = False,
    server_draft_profile_model_calls_per_count: int = 100,
    server_draft_profile_width_list: str = "10,20,30,40,50,60,70,80,90,100,110,120,130,140,150",
):
    """Load profiling data (automatically generated when draft is missing)."""
    profile_wall_start = time.monotonic()
    profile_timing_summary: Dict[str, Any] = {
        "draft_profile_generated": False,
        "draft_profile_loaded": False,
        "draft_profile_wall_sec": 0.0,
        "target_profile_generated": False,
        "target_profile_loaded": False,
        "target_profile_request_wall_sec": 0.0,
        "target_profile_wall_sec": 0.0,
        "target_verification_profile_sec": 0.0,
        "target_server_draft_profile_sec": 0.0,
    }
    if not device_name:
        profile_timing_summary["profile_total_wall_sec"] = float(time.monotonic() - profile_wall_start)
        return profile_timing_summary
    draft_profile_file, target_profile_file = _get_profile_paths(
        draft_model_path=draft_model_path,
        base_model_path=base_model_path,
        bench_name=bench_name,
        device_name=device_name,
        server_name=server_name,
        target_quantization=target_quantization,
        draft_quantization=draft_quantization,
    )
    if not bool(draft_profile_force_refresh):
        resolved_draft_profile_file = _resolve_profile_variant_path(draft_profile_file, "dq")
        if resolved_draft_profile_file != draft_profile_file and debug:
            print(
                "[Profile] draft profile fallback hit: "
                f"requested={draft_profile_file} resolved={resolved_draft_profile_file}"
            )
        draft_profile_file = resolved_draft_profile_file
    if not bool(target_profile_force_refresh):
        resolved_target_profile_file = _resolve_profile_variant_path(target_profile_file, "tq")
        if resolved_target_profile_file != target_profile_file and debug:
            print(
                "[Profile] target profile fallback hit: "
                f"requested={target_profile_file} resolved={resolved_target_profile_file}"
            )
        target_profile_file = resolved_target_profile_file
    runner.draft_profile_file = draft_profile_file
    runner.target_profile_file = target_profile_file
    runner.online_profile_update_enabled = bool(online_profile_update)
    runner.online_profile_lr = _sanitize_online_lr(online_profile_lr)

    # Draft 
    draft_profile_refresh_needed = False
    profile_data = None
    if os.path.exists(draft_profile_file):
        try:
            with open(draft_profile_file, 'r') as f:
                profile_data = json.load(f)
            profile_timing_summary["draft_profile_loaded"] = True
            if debug:
                print(f"Profiling data loaded: {draft_profile_file}")
        except Exception as e:
            if debug:
                print(f"Failed to load profiling data: {e}")
            profile_data = None
    else:
        legacy_draft_profile_file = os.path.join(
            os.path.dirname(draft_profile_file),
            f"{device_name}_{draft_model_path.split('/')[-1] if '/' in draft_model_path else draft_model_path}_draft.json",
        )
        if os.path.exists(legacy_draft_profile_file):
            try:
                with open(legacy_draft_profile_file, 'r') as f:
                    profile_data = json.load(f)
                profile_timing_summary["draft_profile_loaded"] = True
                if debug:
                    print(f"Legacy draft profile loaded: {legacy_draft_profile_file}")
            except Exception:
                profile_data = None

    def _profile_has_gpu_energy_fields(data: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(data, dict) or not data:
            return False
        found = False
        for _width, row in data.items():
            if not isinstance(row, dict):
                continue
            if "model_call_avg_time_ms" not in row:
                continue
            found = True
            if ("gpu_power_avg_w" not in row) or ("gpu_total_energy_kwh" not in row):
                return False
        return found

    if bool(draft_profile_force_refresh):
        print(f"Draft profile force refresh requested: {draft_profile_file}")
        profile_data = None
        draft_profile_refresh_needed = True

    if (
        profile_data is not None
        and runner.uses_draft_energy_profile()
        and (not _profile_has_gpu_energy_fields(profile_data))
    ):
        print(
            "Draft profile lacks GPU energy fields; reprofiling for draft-energy-based objectives: "
            f"{draft_profile_file}"
        )
        profile_data = None
        draft_profile_refresh_needed = True


    # Target 
    target_profile_data = None
    if os.path.exists(target_profile_file):
        try:
            with open(target_profile_file, 'r') as f:
                target_profile_data = json.load(f)
            profile_timing_summary["target_profile_loaded"] = True
            if debug:
                print(f"Target Profiling data loaded: {target_profile_file}")
        except Exception as e:
            if debug:
                print(f"Target Failed to load profiling data: {e}")
            target_profile_data = None
    else:
        legacy_target_profile_file = os.path.join(
            os.path.dirname(target_profile_file),
            f"{server_name}_{base_model_path.split('/')[-1] if '/' in base_model_path else base_model_path}_target.json",
        )
        if os.path.exists(legacy_target_profile_file):
            try:
                with open(legacy_target_profile_file, 'r') as f:
                    target_profile_data = json.load(f)
                profile_timing_summary["target_profile_loaded"] = True
                if debug:
                    print(f"Legacy target profile loaded: {legacy_target_profile_file}")
            except Exception:
                target_profile_data = None

    if bool(target_profile_force_refresh):
        print(f"Target profile force refresh requested: {target_profile_file}")
        target_profile_data = None
        profile_timing_summary["target_profile_loaded"] = False

    def _run_draft_profile_local() -> Optional[Dict[str, Any]]:
        draft_profile_start = time.monotonic()
        width_list = _parse_int_list_csv(draft_profile_width_list, list(DRAFT_TREE_PROFILE_WIDTHS))
        if draft_profile_refresh_needed:
            print(f"Refreshing draft profiling file: {draft_profile_file}")
        else:
            print(f"Draft profiling file is missing. Starting profiling: {draft_profile_file}")
        local_profile = profile_width_timing(
            runner=runner,
            tokenizer=tokenizer,
            max_depth=max_depth,
            draft_model_path=draft_model_path,
            device_name=device_name,
            draft_quantization=draft_quantization,
            question_file=question_file,
            bench_name=bench_name,
            width_list=width_list,
            target_model_calls_per_width=max(1, int(draft_profile_model_calls_per_count)),
            fixed_depth=fixed_depth,
            force_refresh=bool(draft_profile_refresh_needed),
            profile_warmup_runs=PROFILE_WARMUP_RUNS,
            profile_burnin_runs=PROFILE_BURNIN_RUNS,
        )
        if local_profile:
            profile_timing_summary["draft_profile_generated"] = True
            print(f"Draft profiling complete: {len(local_profile)} widths collected")
        profile_timing_summary["draft_profile_wall_sec"] += float(time.monotonic() - draft_profile_start)
        return local_profile

    def _run_target_profile_remote() -> Tuple[bool, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        target_profile_start = time.monotonic()
        width_defaults = [50]
        depth_defaults = [10]
        node_defaults = list(range(10, 151, 10))
        width_list = _parse_int_list_csv(target_profile_width_list, width_defaults)
        depth_list = _parse_int_list_csv(target_profile_depth_list, depth_defaults)
        node_list = _parse_int_list_csv(target_profile_node_list, node_defaults)
        print(f"Target profiling file is missing. Requesting automatic generation: {target_profile_file}")
        ok, data, timing = _request_target_profile_generation(
            target_sock,
            runner=runner,
            tokenizer=tokenizer,
            base_model_path=base_model_path,
            draft_model_path=draft_model_path,
            bench_name=bench_name,
            question_file=question_file,
            draft_quantization=draft_quantization,
            device_map=draft_device_map,
            width_list=width_list,
            depth_list=depth_list,
            node_list=node_list,
            runs_per_combination=int(target_profile_model_calls_per_count),
            force_refresh=bool(target_profile_force_refresh),
            poll_interval_sec=float(target_profile_poll_interval_sec),
            timeout_sec=float(target_profile_timeout_sec),
            debug=bool(debug),
        )
        profile_timing_summary["target_profile_request_wall_sec"] += float(time.monotonic() - target_profile_start)
        return ok, data, timing

    def _apply_target_profile_result(
        ok: bool,
        transferred_target_profile_data: Optional[Dict[str, Any]],
        transferred_timing_summary: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        local_target_profile_data = None
        if isinstance(transferred_timing_summary, dict):
            wall_sec = float(transferred_timing_summary.get("target_profile_wall_sec", 0.0) or 0.0)
            verification_sec = float(transferred_timing_summary.get("target_verification_profile_sec", 0.0) or 0.0)
            server_draft_sec = float(transferred_timing_summary.get("target_server_draft_profile_sec", 0.0) or 0.0)
            profile_timing_summary["target_profile_wall_sec"] = wall_sec
            profile_timing_summary["target_verification_profile_sec"] = verification_sec
            profile_timing_summary["target_server_draft_profile_sec"] = server_draft_sec
            print(
                "Target profiling time summary: "
                f"wall_sec={wall_sec:.6f} "
                f"verification_sec={verification_sec:.6f} "
                f"server_draft_sec={server_draft_sec:.6f}"
            )
        if ok and isinstance(transferred_target_profile_data, dict) and transferred_target_profile_data:
            try:
                os.makedirs(os.path.dirname(target_profile_file), exist_ok=True)
                with open(target_profile_file, "w", encoding="utf-8") as f:
                    json.dump(transferred_target_profile_data, f, indent=2, ensure_ascii=False)
                local_target_profile_data = transferred_target_profile_data
                profile_timing_summary["target_profile_generated"] = True
                print(f"Target profiling auto-generation/transfer/save complete: {target_profile_file}")
            except Exception as e:
                print(f"Failed to save transferred target profile: {e}")
        elif ok and os.path.exists(target_profile_file):
            try:
                with open(target_profile_file, "r") as f:
                    local_target_profile_data = json.load(f)
                profile_timing_summary["target_profile_loaded"] = True
                print(f"Target profiling auto-generation/load complete: {target_profile_file}")
            except Exception as e:
                if debug:
                    print(f"Failed to load target auto-generated file: {e}")
        elif ok:
            print(f"Received target profiling auto-generation completion response, but file is missing: {target_profile_file}")
        else:
            print(f"Target profiling auto-generation failed: {target_profile_file}")
        return local_target_profile_data

    need_draft_profile = (profile_data is None)
    need_target_profile = (target_profile_data is None and auto_target_profile and target_sock is not None)

    # ( draft + target RPC)
    if need_draft_profile and need_target_profile:
        target_result_holder: Dict[str, Any] = {"ok": False, "data": None, "timing": None}

        def _target_job():
            try:
                ok, data, timing = _run_target_profile_remote()
                target_result_holder["ok"] = bool(ok)
                target_result_holder["data"] = data
                target_result_holder["timing"] = timing
            except Exception as e:
                if debug:
                    print(f"[TargetProfile][ERROR] parallel target job failed: {e}")
                target_result_holder["ok"] = False
                target_result_holder["data"] = None
                target_result_holder["timing"] = None

        target_thread = threading.Thread(target=_target_job, daemon=True)
        target_thread.start()

        profile_data = _run_draft_profile_local()

        target_thread.join()
        target_profile_data = _apply_target_profile_result(
            ok=bool(target_result_holder.get("ok", False)),
            transferred_target_profile_data=target_result_holder.get("data"),
            transferred_timing_summary=target_result_holder.get("timing"),
        )
    else:
        if need_draft_profile:
            profile_data = _run_draft_profile_local()

        if target_profile_data is None:
            if auto_target_profile and target_sock is not None:
                ok, transferred_target_profile_data, transferred_timing_summary = _run_target_profile_remote()
                target_profile_data = _apply_target_profile_result(
                    ok=ok,
                    transferred_target_profile_data=transferred_target_profile_data,
                    transferred_timing_summary=transferred_timing_summary,
                )
            else:
                if auto_target_profile and target_sock is None:
                    print(
                        f"Target profiling file is missing. "
                        f"Will attempt auto-generation after target connection: {target_profile_file}"
                    )
                else:
                    print(
                        f"Target profiling file is missing. "
                        f"Target profiling must be run separately: {target_profile_file}"
                    )

    # DraftRunner
    runner.profile_data = profile_data
    runner.target_profile_data = target_profile_data
    profile_timing_summary["profile_total_wall_sec"] = float(time.monotonic() - profile_wall_start)
    print(
        "[ProfileTiming] "
        f"total_sec={profile_timing_summary['profile_total_wall_sec']:.6f} "
        f"draft_sec={profile_timing_summary['draft_profile_wall_sec']:.6f} "
        f"target_request_sec={profile_timing_summary['target_profile_request_wall_sec']:.6f} "
        f"target_wall_sec={profile_timing_summary['target_profile_wall_sec']:.6f}"
    )
    return profile_timing_summary

def run_draft(
    host: str,
    port: int,
    base_model_path: str,
    draft_model_path: str,
    bench_name: str,
    question_file: str,
    limit: int,
    temperature: float,
    nodes: int,
    max_depth: int,
    device_map: str,
    load_in_4bit: bool,
    load_in_8bit: bool,
    target_quantization: str = "8bit",
    answer_file: str = None,
    model_id: str = "llama-2-chat",
    num_choices: int = 1,
    debug: bool = False,
    print_tree: bool = False,
    per_token_probability_bound: float = 0.0,
    per_path_probability_bound: float = 0.0,
    device_name: str = None,
    draft_per_hour_cost: float = 0.0,
    target_per_hour_cost: float = 1.208,
    draft_electricity_cost_per_kwh: float = 0.2,
    user_communication_cost_per_gb: float = None,
    cloud_outbound_cost_per_gb: float = None,
    accept_length_margin: float = 0.05,
    objective_selection_mode: str = "blend",
    constraint_target: str = "metric",
    metric_constraint_per_1m_token: float = None,
    min_tps_constraint: float = None,
    total_metric_cap: float = float("inf"),
    cost_sensitivity: float = 0.0,
    opt_tree: bool = False,
    model_family: str = "llama2",
    min_width: int = 1,
    fixed_depth: bool = False,
    fixed_nnodes: bool = False,
    fixed_width: bool = False,
    fixed_width_value: int = None,
    server_name: str = "rtxproa6000",
    enable_gpu_monitor: bool = False,
    gpu_monitor_interval: float = 0.05,
    enable_cpu_monitor: bool = False,
    fix_gpu_clock: bool = False,
    gpu_graphics_clock_mhz: int = None,
    gpu_memory_clock_mhz: int = None,
    proactive_drafting: bool = True,
    proactive_threshold: float = 0.0,
    adaptive_proactive_threshold: bool = False,
    disable_proactive_budget: bool = False,
    join_canceled_proactive_before_tree_build: bool = False,
    no_draft_cost: bool = False,
    objective_metric: str = "total_cost",
    server_only_baseline_json: str = None,
    disable_server_only: bool = False,
    force_server_only: bool = False,
    force_server_only_ar: bool = False,
    server_only_ar_turn_rpc: bool = True,
    server_only_ar_max_new_tokens: int = 256,
    max_new_tokens: int = 256,
    reference_test_mode: bool = False,
    reference_test_runs: int = 5,
    reference_test_output_json: str = None,
    reference_cs_curve_rounds: int = 1,
    reference_max_steps_limit: int = 1,
    reference_constraint_multipliers: str = "0.8,1.0,1.2",
    reference_force_refresh: bool = False,
    tokenizer_path: str = None,
    auto_target_profile: bool = True,
    draft_profile_force_refresh: bool = False,
    draft_profile_model_calls_per_count: int = 100,
    draft_profile_width_list: str = "10,20,30,40,50,60,70,80,90,100,110,120,130,140,150",
    draft_profile_only: bool = False,
    target_profile_force_refresh: bool = False,
    target_profile_model_calls_per_count: int = 20,
    target_profile_width_list: str = "50",
    target_profile_depth_list: str = "10",
    target_profile_node_list: str = "10,20,30,40,50,60,70,80,90,100,110,120,130,140,150",
    profile_only: bool = False,
    profile_only_report_json: str = None,
    reference_only_exit_after_cache: bool = False,
    online_profile_update: bool = True,
    online_profile_lr: float = ONLINE_PROFILE_LR_DEFAULT,
    accept_length_calibration: bool = True,
    target_time_calibration: bool = True,
    bill_draft_as_target_gpu: bool = False,
    server_draft_profile_auto: bool = True,
    server_draft_profile_force_refresh: bool = False,
    server_draft_profile_model_calls_per_count: int = 100,
    server_draft_profile_width_list: str = "10,20,30,40,50,60,70,80,90,100,110,120,130,140,150",
):
    objective_metric = _normalize_objective_metric(objective_metric)
    max_new_tokens = max(1, int(max_new_tokens))
    server_only_ar_max_new_tokens = max(1, int(server_only_ar_max_new_tokens))
    if (
        (force_server_only or force_server_only_ar)
        and objective_metric in {"total_cost", "api_cost"}
        and (not bool(no_draft_cost))
        and (not bool(bill_draft_as_target_gpu))
    ):
        print(
            "[INFO] force_server_only/force_server_only_ar + cost objective -> enabling time-based draft billing on target "
            "(draft billed as target_per_sec_cost)."
        )
        bill_draft_as_target_gpu = True
    cost_objective_metrics = {"total_cost", "api_cost"}
    energy_objective_metrics = {"draft_energy", "target_energy"}
    if objective_metric not in {*cost_objective_metrics, *energy_objective_metrics}:
        raise ValueError(
            "Unsupported objective_metric: "
            f"{objective_metric}. Use 'total_cost', 'api_cost', 'draft_energy', or 'target_energy'."
        )
    if objective_metric == "draft_energy" and not enable_gpu_monitor:
        print(f"[INFO] objective_metric={objective_metric} -> enabling GPU monitor automatically.")
        enable_gpu_monitor = True
    if objective_metric == "target_energy" and not enable_gpu_monitor:
        print("[INFO] target_energy objective -> enabling GPU monitor (server-only uses draft+target energy sum).")
        enable_gpu_monitor = True
    if objective_metric == "total_cost" and (not no_draft_cost) and (not bill_draft_as_target_gpu) and not enable_gpu_monitor:
        print("[INFO] objective_metric=total_cost -> enabling GPU monitor (required for GPU-energy-based draft cost).")
        enable_gpu_monitor = True
    user_communication_cost_per_gb, cloud_outbound_cost_per_gb = _resolve_cloud_transfer_costs(
        user_communication_cost_per_gb=user_communication_cost_per_gb,
        cloud_outbound_cost_per_gb=cloud_outbound_cost_per_gb,
    )
    objective_selection_mode = str(objective_selection_mode).lower()
    if objective_selection_mode not in {"blend", "constraint"}:
        raise ValueError(
            f"Unsupported objective_selection_mode: {objective_selection_mode}. Use 'blend' or 'constraint'."
        )
    constraint_target = str(constraint_target).lower()
    if constraint_target not in {"metric", "tps"}:
        raise ValueError(
            f"Unsupported constraint_target: {constraint_target}. Use 'metric' or 'tps'."
        )
    if objective_selection_mode == "constraint":
        if constraint_target == "metric" and metric_constraint_per_1m_token is not None and float(metric_constraint_per_1m_token) <= 0:
            raise ValueError(
                "--metric-constraint-per-1m-token must be > 0 when provided."
            )
        if constraint_target == "tps" and min_tps_constraint is not None and float(min_tps_constraint) < 0:
            raise ValueError("--min-tps-constraint must be >= 0 when provided.")
    user_metric_constraint_provided = (
        constraint_target == "metric" and metric_constraint_per_1m_token is not None
    )
    metric_constraint_per_token = (
        (float(metric_constraint_per_1m_token) / 1_000_000.0)
        if constraint_target == "metric" and metric_constraint_per_1m_token is not None
        else None
    )
    total_metric_cap = float(total_metric_cap) if total_metric_cap is not None else float("inf")
    if np.isfinite(total_metric_cap) and total_metric_cap <= 0:
        raise ValueError("--total-metric-cap must be > 0 when provided.")
    reference_cs_curve_rounds = max(1, int(reference_cs_curve_rounds))
    reference_max_steps_limit = max(1, int(reference_max_steps_limit))
    reference_constraint_multipliers_list = _parse_reference_constraint_multipliers(
        reference_constraint_multipliers
    )
    if objective_selection_mode == "constraint":
        reference_mode_key = (
            f"{objective_selection_mode}|{constraint_target}|{objective_metric}|auto-center-blendcs50|"
            + ",".join(f"{v:.6f}" for v in reference_constraint_multipliers_list)
        )
    else:
        reference_mode_key = (
            f"{objective_selection_mode}|{constraint_target}|{objective_metric}|"
            + ",".join(f"{v:.6f}" for v in reference_constraint_multipliers_list)
        )
    accept_length_margin = max(0.0, min(0.99, float(accept_length_margin)))
    if not server_only_baseline_json:
        script_dir = os.path.dirname(__file__)
        parent_dir = os.path.dirname(script_dir)
        server_only_baseline_json = os.path.join(_resolve_data_root(parent_dir), "profile", "server_only_sd_baseline.json")
    requested_draft_quantization = (
        "8bit" if bool(load_in_8bit) else ("4bit" if bool(load_in_4bit) else "none")
    )
    draft_quant_fallback_chain = _build_draft_quantization_fallback_chain(
        requested_draft_quantization,
        load_in_4bit=bool(load_in_4bit),
        load_in_8bit=bool(load_in_8bit),
    )
    KVLlamaForCausalLM = get_kv_llama_class(base_model_path)
    draft_model = None
    draft_quantization = requested_draft_quantization
    attempted_draft_quants: List[str] = []
    draft_failures: List[str] = []
    for quant_mode in draft_quant_fallback_chain:
        attempted_draft_quants.append(str(quant_mode))
        quantization_config = _build_quantization_config_for_mode(quant_mode)
        try:
            draft_model = KVLlamaForCausalLM.from_pretrained(
                draft_model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map=device_map,
                quantization_config=quantization_config,
                token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None,
            )
            draft_quantization = str(quant_mode)
            break
        except Exception as e:
            draft_failures.append(f"{quant_mode}:{e}")
            _release_partial_draft_model(draft_model)
            draft_model = None
            if not _is_memory_related_load_error(str(e)):
                break
    if draft_model is None:
        failures_text = " | ".join(draft_failures) if draft_failures else "unknown_failure"
        attempted_text = " -> ".join(attempted_draft_quants) if attempted_draft_quants else "none"
        raise RuntimeError(
            "Failed to load draft model after quantization fallback. "
            f"attempted={attempted_text}; failures={failures_text}. "
            "Try a smaller draft model or start with higher quantization."
        )
    if len(attempted_draft_quants) > 1 or draft_quantization != requested_draft_quantization:
        print(
            "[Draft Fallback] loaded with quantization="
            f"{draft_quantization} (requested={requested_draft_quantization}, "
            f"attempted={attempted_draft_quants})"
        )
    tokenizer_source = tokenizer_path or base_model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None,
    )
    # total_cost draft GPU (kWh) * ($/kWh) .
    # , bill_draft_as_target_gpu draft target_per_sec_cost .
    target_per_sec_cost = target_per_hour_cost / 3600.0
    if objective_metric == "total_cost":
        if float(draft_per_hour_cost or 0.0) > 0:
            print(
                "[WARN] --draft-per-hour-cost is ignored for objective_metric=total_cost. "
                "Use --draft-electricity-cost-per-kwh instead."
            )
        draft_per_sec_cost = 0.0
    else:
        draft_per_sec_cost = draft_per_hour_cost / 3600.0
    if bool(bill_draft_as_target_gpu) and (not bool(no_draft_cost)):
        draft_per_sec_cost = float(target_per_sec_cost)
    runner = DraftRunner(
        draft_model=draft_model,
        tokenizer=tokenizer,
        debug=debug,
        draft_per_sec_cost=draft_per_sec_cost,
        target_per_sec_cost=target_per_sec_cost,
        draft_electricity_cost_per_kwh=draft_electricity_cost_per_kwh,
        user_communication_cost_per_gb=user_communication_cost_per_gb,
        cloud_outbound_cost_per_gb=cloud_outbound_cost_per_gb,
        cost_sensitivity=cost_sensitivity,
        enable_gpu_monitor=enable_gpu_monitor,
        gpu_monitor_interval=gpu_monitor_interval,
        enable_cpu_monitor=enable_cpu_monitor,
        fix_gpu_clock=bool(fix_gpu_clock),
        gpu_graphics_clock_mhz=gpu_graphics_clock_mhz,
        gpu_memory_clock_mhz=gpu_memory_clock_mhz,
        opt_tree=opt_tree,
        no_draft_cost=no_draft_cost,
        objective_metric=objective_metric,
        accept_length_margin=accept_length_margin,
        objective_selection_mode=objective_selection_mode,
        constraint_target=constraint_target,
        metric_constraint_per_token=metric_constraint_per_token,
        min_tps_constraint=min_tps_constraint,
        bill_draft_as_target_gpu=bool(bill_draft_as_target_gpu),
    )
    if objective_metric == "total_cost" and (not no_draft_cost) and (not bill_draft_as_target_gpu):
        if runner.gpu_monitor is None:
            raise RuntimeError(
                "total_cost requires GPU monitor for draft energy cost, but GPU monitor is not initialized."
            )
        try:
            gpu_info = runner.gpu_monitor.get_gpu_info()
        except Exception as e:
            raise RuntimeError(
                f"total_cost requires GPU monitor, but failed to query GPU info: {e}"
            ) from e
        if not gpu_info:
            raise RuntimeError(
                "total_cost requires GPU monitor, but no GPU device info is available."
            )

    enable_target_profile_auto = bool(auto_target_profile) and (not bool(draft_profile_only))
    draft_profile_force_refresh_consumed = False
    # (device_name )
    chat_draft_profile_force_refresh_consumed = False
    profile_timing_summaries: List[Dict[str, Any]] = []
    initial_profile_timing_summary = load_profile_data(
        runner=runner,
        tokenizer=tokenizer,
        max_depth=max_depth,
        draft_model_path=draft_model_path,
        base_model_path=base_model_path,
        bench_name=bench_name,
        device_name=device_name,
        server_name=server_name,
        target_quantization=target_quantization,
        draft_quantization=draft_quantization,
        question_file=question_file,
        fixed_depth=fixed_depth,
        debug=debug,
        auto_target_profile=enable_target_profile_auto,
        draft_profile_force_refresh=bool(draft_profile_force_refresh),
        draft_profile_model_calls_per_count=max(1, int(draft_profile_model_calls_per_count)),
        draft_profile_width_list=draft_profile_width_list,
        online_profile_update=bool(online_profile_update),
        online_profile_lr=float(online_profile_lr),
    )
    if isinstance(initial_profile_timing_summary, dict):
        profile_timing_summaries.append(initial_profile_timing_summary)
    if bool(draft_profile_force_refresh):
        chat_draft_profile_force_refresh_consumed = True
    if bool(draft_profile_force_refresh):
        draft_profile_force_refresh_consumed = True

    draft_profile_file, target_profile_file = _get_profile_paths(
        draft_model_path=draft_model_path,
        base_model_path=base_model_path,
        bench_name=bench_name,
        device_name=device_name,
        server_name=server_name,
        target_quantization=target_quantization,
        draft_quantization=draft_quantization,
    )

    if profile_only and bool(draft_profile_only):
        profile_only_summary = {
            "mode": "profile_only",
            "draft_profile_only": True,
            "bench_name": bench_name,
            "base_model_path": base_model_path,
            "draft_model_path": draft_model_path,
            "device_name": device_name,
            "server_name": server_name,
            "target_quantization": target_quantization,
            "draft_quantization": draft_quantization,
            "draft_profile_file": draft_profile_file,
            "target_profile_file": target_profile_file,
            "draft_profile_exists": bool(draft_profile_file and os.path.exists(draft_profile_file)),
            "target_profile_exists": bool(target_profile_file and os.path.exists(target_profile_file)),
            "target_profile_skipped": True,
            "draft_profile_force_refresh": bool(draft_profile_force_refresh),
            "draft_profile_model_calls_per_count": int(max(1, int(draft_profile_model_calls_per_count))),
            "target_profile_force_refresh": bool(target_profile_force_refresh),
            "target_profile_model_calls_per_count": int(target_profile_model_calls_per_count),
            "profile_timing_summaries": profile_timing_summaries,
            "draft_gpu_metric_missing_summary": _summarize_draft_profile_missing_stats(draft_profile_file),
        }
        print(
            "[ProfileOnly] draft-only done: "
            f"draft_profile_exists={profile_only_summary['draft_profile_exists']} "
            f"target_profile_skipped={profile_only_summary['target_profile_skipped']}"
        )
        if profile_only_report_json:
            report_path = os.path.expanduser(profile_only_report_json)
            report_dir = os.path.dirname(report_path)
            if report_dir:
                os.makedirs(report_dir, exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(profile_only_summary, f, indent=2, ensure_ascii=False)
            print(f"[ProfileOnly] report saved: {report_path}")
        return profile_only_summary

    with _connect_target_with_retry(
        host=host,
        port=port,
        max_attempts=int(os.environ.get("AUTODRAFT_DRAFT_CONNECT_RETRIES", "8")),
        base_delay_s=float(os.environ.get("AUTODRAFT_DRAFT_CONNECT_BACKOFF_SEC", "0.5")),
        connect_timeout_s=float(os.environ.get("AUTODRAFT_DRAFT_CONNECT_TIMEOUT_SEC", "10.0")),
    ) as sock:
        target_sync_info = _ensure_remote_target_model(
            sock=sock,
            base_model_path=base_model_path,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            target_quantization=target_quantization,
            device_map=device_map,
            debug=debug,
        )
        resolved_target_quantization = _normalize_quantization_mode(
            target_sync_info.get("selected_quantization", target_quantization), default=str(target_quantization or "8bit")
        )
        if debug and bool(target_sync_info.get("fallback_applied", False)):
            print(
                "[Target Sync] quantization fallback applied: "
                f"attempted={target_sync_info.get('attempted_quantizations')} "
                f"selected={resolved_target_quantization}"
            )
        # Target profile target RPC / .
        connected_profile_timing_summary = load_profile_data(
            runner=runner,
            tokenizer=tokenizer,
            max_depth=max_depth,
            draft_model_path=draft_model_path,
            base_model_path=base_model_path,
            bench_name=bench_name,
            device_name=device_name,
            server_name=server_name,
            target_quantization=resolved_target_quantization,
            draft_quantization=draft_quantization,
            question_file=question_file,
            fixed_depth=fixed_depth,
            debug=debug,
            target_sock=sock,
            auto_target_profile=enable_target_profile_auto,
            draft_profile_force_refresh=bool(draft_profile_force_refresh)
            and not bool(draft_profile_force_refresh_consumed),
            draft_profile_model_calls_per_count=max(1, int(draft_profile_model_calls_per_count)),
            draft_profile_width_list=draft_profile_width_list,
            target_profile_force_refresh=bool(target_profile_force_refresh),
            target_profile_model_calls_per_count=int(target_profile_model_calls_per_count),
            target_profile_width_list=target_profile_width_list,
            target_profile_depth_list=target_profile_depth_list,
            target_profile_node_list=target_profile_node_list,
            draft_device_map=device_map,
            online_profile_update=bool(online_profile_update),
            online_profile_lr=float(online_profile_lr),
        )
        if isinstance(connected_profile_timing_summary, dict):
            profile_timing_summaries.append(connected_profile_timing_summary)
        draft_profile_file = str(runner.draft_profile_file or draft_profile_file)
        target_profile_file = str(runner.target_profile_file or target_profile_file)
        if profile_only:
            profile_only_summary = {
                "mode": "profile_only",
                "draft_profile_only": bool(draft_profile_only),
                "bench_name": bench_name,
                "base_model_path": base_model_path,
                "draft_model_path": draft_model_path,
                "device_name": device_name,
                "server_name": server_name,
                "target_quantization": resolved_target_quantization,
                "draft_quantization": draft_quantization,
                "draft_profile_file": draft_profile_file,
                "target_profile_file": target_profile_file,
                "draft_profile_exists": bool(draft_profile_file and os.path.exists(draft_profile_file)),
                "target_profile_exists": bool(target_profile_file and os.path.exists(target_profile_file)),
                "target_profile_skipped": False,
                "draft_profile_force_refresh": bool(draft_profile_force_refresh),
                "draft_profile_model_calls_per_count": int(max(1, int(draft_profile_model_calls_per_count))),
                "target_profile_force_refresh": bool(target_profile_force_refresh),
                "target_profile_model_calls_per_count": int(target_profile_model_calls_per_count),
                "profile_timing_summaries": profile_timing_summaries,
                "draft_gpu_metric_missing_summary": _summarize_draft_profile_missing_stats(draft_profile_file),
            }
            print(
                "[ProfileOnly] done: "
                f"draft_profile_exists={profile_only_summary['draft_profile_exists']} "
                f"target_profile_exists={profile_only_summary['target_profile_exists']}"
            )
            if profile_only_report_json:
                report_path = os.path.expanduser(profile_only_report_json)
                report_dir = os.path.dirname(report_path)
                if report_dir:
                    os.makedirs(report_dir, exist_ok=True)
                with open(report_path, "w", encoding="utf-8") as f:
                    json.dump(profile_only_summary, f, indent=2, ensure_ascii=False)
                print(f"[ProfileOnly] report saved: {report_path}")
            return profile_only_summary
        questions = _load_benchmark_questions(bench_name, question_file)
        if limit and limit > 0:
            questions = questions[:limit]

        if bool(reference_force_refresh):
            _, reference_cache_path = _reference_cache_paths(
                base_model_path=base_model_path,
                draft_model_path=draft_model_path,
                bench_name=bench_name,
                objective_metric=objective_metric,
                server_name=server_name,
                device_name=device_name,
                target_quantization=resolved_target_quantization,
                draft_quantization=draft_quantization,
                objective_selection_mode=objective_selection_mode,
                reference_mode_key=reference_mode_key,
            )
            reference_cache = None
            print(
                "[Reference] force refresh enabled; ignoring existing cache and rebuilding: "
                f"{reference_cache_path}"
            )
        else:
            reference_cache, reference_cache_path = load_reference_cache(
                base_model_path=base_model_path,
                draft_model_path=draft_model_path,
                bench_name=bench_name,
                objective_metric=objective_metric,
                server_name=server_name,
                device_name=device_name,
                target_quantization=resolved_target_quantization,
                draft_quantization=draft_quantization,
                objective_selection_mode=objective_selection_mode,
                reference_mode_key=reference_mode_key,
            )
        reference_feasible_metric_per_token = None
        reference_feasible_tps = None
        reference_cs_anchor_curve = None
        reference_tradeoff_curve_cs0_1 = None
        reference_constraint_anchor_curve = None
        reference_tradeoff_curve_by_constraint = None
        auto_reference_constraint_center_per_1m = None
        auto_reference_constraint_center_tps = None
        if bool(force_server_only_ar):
            print("[Warmup] Skipping speculative warmup for server-only AR mode.")
            if reference_cache is None:
                point_metric = (
                    float(runner.reference_cost_per_token)
                    if objective_metric == "total_cost"
                    else float(runner.reference_objective_per_token)
                )
                reference_cache = {
                    "reference_tps": float(runner.reference_tps),
                    "reference_cost_per_token": float(runner.reference_cost_per_token),
                    "reference_objective_per_token": float(runner.reference_objective_per_token),
                    "feasible_metric_per_token": {
                        "min": point_metric,
                        "max": point_metric,
                        "mean": point_metric,
                        "count": 1,
                    },
                    "feasible_tps": {
                        "min": float(runner.reference_tps),
                        "max": float(runner.reference_tps),
                        "mean": float(runner.reference_tps),
                        "count": 1,
                    },
                }
        else:
            print("[Warmup] Running warmup (3 rounds)")
            run_warmup(
                sock=sock,
                runner=runner,
                tokenizer=tokenizer,
                questions=questions,
                base_model_path=base_model_path,
                bench_name=bench_name,
                nodes=nodes,
                max_depth=max_depth,
                per_token_probability_bound=per_token_probability_bound,
                per_path_probability_bound=per_path_probability_bound,
                min_width=min_width,
                fixed_width=fixed_width,
                fixed_width_value=fixed_width_value,
                fixed_nnodes=fixed_nnodes,
                fixed_depth=fixed_depth,
                debug=debug,
                warmup_cost_sensitivity=0.5,
                warmup_rounds=3,
                full_query=False,
            )
        if reference_cache is not None:
            runner.reference_tps = float(reference_cache.get("reference_tps", runner.reference_tps))
            runner.reference_cost_per_token = float(
                reference_cache.get("reference_cost_per_token", runner.reference_cost_per_token)
            )
            runner.reference_objective_per_token = float(
                reference_cache.get("reference_objective_per_token", runner.reference_objective_per_token)
            )
            reference_draft_objective_rate_per_sec = float(
                reference_cache.get("reference_draft_objective_rate_per_sec", 0.0) or 0.0
            )
            reference_target_objective_rate_per_sec = float(
                reference_cache.get("reference_target_objective_rate_per_sec", 0.0) or 0.0
            )
            if runner.uses_draft_energy_objective():
                if reference_draft_objective_rate_per_sec > 0:
                    runner.draft_objective_rate_per_sec = reference_draft_objective_rate_per_sec
            if runner.uses_target_energy_objective():
                if reference_target_objective_rate_per_sec > 0:
                    runner.target_objective_rate_per_sec = reference_target_objective_rate_per_sec
            reference_feasible_metric_per_token = reference_cache.get("feasible_metric_per_token", None)
            reference_feasible_tps = reference_cache.get("feasible_tps", None)
            reference_cs_anchor_curve = reference_cache.get("reference_cs_anchor_curve", None)
            reference_tradeoff_curve_cs0_1 = reference_cache.get("reference_tradeoff_curve_cs0_1", None)
            reference_constraint_anchor_curve = reference_cache.get("reference_constraint_anchor_curve", None)
            reference_tradeoff_curve_by_constraint = reference_cache.get("reference_tradeoff_curve_by_constraint", None)
            auto_reference_constraint_center_per_1m = reference_cache.get(
                "reference_constraint_center_per_1m_token", None
            )
            selected_ref = _select_reference_anchor_for_cs(
                reference_cs_anchor_curve,
                float(runner.cost_sensitivity),
            )
            if isinstance(selected_ref, dict):
                selected_tps = float(selected_ref.get("predicted_tps", 0.0) or 0.0)
                selected_cost_per_token = float(selected_ref.get("predicted_cost_per_token", 0.0) or 0.0)
                selected_obj_per_token = float(selected_ref.get("predicted_objective_per_token", 0.0) or 0.0)
                if selected_tps > 0:
                    runner.reference_tps = selected_tps
                if selected_cost_per_token > 0:
                    runner.reference_cost_per_token = selected_cost_per_token
                if selected_obj_per_token > 0:
                    runner.reference_objective_per_token = selected_obj_per_token
                selected_draft_rate = float(selected_ref.get("draft_objective_rate_per_sec", 0.0) or 0.0)
                selected_target_rate = float(selected_ref.get("target_objective_rate_per_sec", 0.0) or 0.0)
                if runner.uses_draft_energy_objective() and selected_draft_rate > 0:
                    runner.draft_objective_rate_per_sec = selected_draft_rate
                if runner.uses_target_energy_objective() and selected_target_rate > 0:
                    runner.target_objective_rate_per_sec = selected_target_rate
                if bool(selected_ref.get("_selected_ref_interpolated", False)):
                    print(
                        "[Reference] CS-specific reference selected "
                        f"(cs={float(runner.cost_sensitivity):.2g}, "
                        f"interp={float(selected_ref.get('_selected_ref_cs_left', 0.0)):.2g}"
                        f"~{float(selected_ref.get('_selected_ref_cs_right', 0.0)):.2g})."
                    )
                else:
                    print(
                        "[Reference] CS-specific reference selected "
                        f"(cs={float(selected_ref.get('_selected_ref_cs', runner.cost_sensitivity)):.2g})."
                    )
            print(f"[Reference] Loaded cached reference: {reference_cache_path}")
        else:
            reference_draft_objective_rate_per_sec = 0.0
            reference_target_objective_rate_per_sec = 0.0
            # Reference anchors should cover at least the first 20 questions by default.
            # (diverse_questions_per_round=True with question_start_index=0)
            reference_point_repeats = max(20, int(reference_cs_curve_rounds))
            fixed_reference_accept_scale = None

            def _run_reference_warmup(
                cost_sensitivity_value: float,
                repeats: int,
                select_last: bool,
                fixed_scale: Optional[float],
                update_ratio: bool,
                max_steps_limit: Optional[int],
                reference_token_count_mode: str = "actual",
                diverse_queries_per_round: bool = False,
                aggregate_query_means: bool = False,
            ) -> dict:
                return run_warmup(
                    sock=sock,
                    runner=runner,
                    tokenizer=tokenizer,
                    questions=questions,
                    base_model_path=base_model_path,
                    bench_name=bench_name,
                    nodes=nodes,
                    max_depth=max_depth,
                    per_token_probability_bound=per_token_probability_bound,
                    per_path_probability_bound=per_path_probability_bound,
                    min_width=min_width,
                    fixed_width=fixed_width,
                    fixed_width_value=fixed_width_value,
                    fixed_nnodes=fixed_nnodes,
                    fixed_depth=fixed_depth,
                    debug=debug,
                    warmup_cost_sensitivity=float(cost_sensitivity_value),
                    warmup_rounds=max(1, int(repeats)),
                    full_query=True,
                    max_steps_per_round=max_steps_limit,
                    select_last_trial=bool(select_last),
                    fixed_accept_length_scale=(
                        float(fixed_scale) if fixed_scale is not None else None
                    ),
                    update_accept_length_ratio=bool(update_ratio),
                    preserve_accept_length_ratio_stats=bool(update_ratio),
                    reference_token_count_mode=str(reference_token_count_mode),
                    diverse_questions_per_round=bool(diverse_queries_per_round),
                    question_start_index=0,
                    aggregate_round_metrics_mean=bool(aggregate_query_means),
                )

            def _build_anchor_row(anchor_value: float, anchor_metrics: dict, fixed_scale: Optional[float]) -> dict:
                anchor_tps = float(anchor_metrics.get("token_per_second", 0.0))
                anchor_cost_per_token = float(anchor_metrics.get("cost_per_token", 0.0))
                anchor_obj_per_token = float(anchor_metrics.get("objective_per_token", 0.0))
                anchor_rate = float(anchor_metrics.get("draft_objective_rate_per_sec", 0.0))
                anchor_target_rate = float(anchor_metrics.get("target_objective_rate_per_sec", 0.0))
                anchor_debug_trace = anchor_metrics.get("reference_debug_trace", None)
                return {
                    "cost_sensitivity": float(anchor_value),
                    "predicted_tps": anchor_tps,
                    "predicted_cost_per_token": anchor_cost_per_token,
                    "predicted_cost_per_1m_token": anchor_cost_per_token * 1_000_000.0,
                    "predicted_objective_per_token": anchor_obj_per_token,
                    "predicted_objective_per_1m_token": anchor_obj_per_token * 1_000_000.0,
                    "predicted_metric_per_token": anchor_obj_per_token,
                    "predicted_metric_per_1m_token": anchor_obj_per_token * 1_000_000.0,
                    "predicted_feasible_metric_per_token": anchor_metrics.get("feasible_metric_per_token", None),
                    "predicted_feasible_tps": anchor_metrics.get("feasible_tps", None),
                    "draft_objective_rate_per_sec": anchor_rate,
                    "target_objective_rate_per_sec": anchor_target_rate,
                    "trial_repeat_count": int(anchor_metrics.get("trial_repeat_count", 1)),
                    "trial_selection_rule": str(anchor_metrics.get("trial_selection_rule", "last_trial")),
                    "reference_token_count_mode": str(anchor_metrics.get("reference_token_count_mode", "actual")),
                    "total_new_tokens_actual": anchor_metrics.get("total_new_tokens_actual", None),
                    "total_new_tokens_for_metric": anchor_metrics.get("total_new_tokens_for_metric", None),
                    "effective_accept_length_for_metric_last": anchor_metrics.get("effective_accept_length_for_metric_last", None),
                    "actual_accept_length_last": anchor_metrics.get("actual_accept_length_last", None),
                    "expected_accept_length_raw_last": anchor_metrics.get("expected_accept_length_raw_last", None),
                    "expected_accept_length_scaled_last": anchor_metrics.get("expected_accept_length_scaled_last", None),
                    "expected_accept_length_clipped_last": anchor_metrics.get("expected_accept_length_clipped_last", None),
                    "accept_length_error_last": anchor_metrics.get("accept_length_error_last", None),
                    "accept_length_error_ratio_last": anchor_metrics.get("accept_length_error_ratio_last", None),
                    "fixed_accept_length_scale": (
                        float(fixed_scale) if fixed_scale is not None else None
                    ),
                    "reference_debug_trace": anchor_debug_trace,
                }
            if objective_selection_mode == "constraint":
                print("[Reference] No cache found. Running full-query constraint anchors independently.")
                print("[Reference] Deriving constraint center from blend(cs=0.5) full-query probe.")
                saved_mode = runner.objective_selection_mode
                saved_constraint_target = getattr(runner, "constraint_target", "metric")
                saved_constraint = runner.metric_constraint_per_token
                saved_min_tps = runner.min_tps_constraint
                try:
                    runner.objective_selection_mode = "blend"
                    runner.metric_constraint_per_token = None
                    runner.min_tps_constraint = None
                    # 1) center calibration: dynamic scale
                    center_metrics = _run_reference_warmup(
                        cost_sensitivity_value=0.5,
                        repeats=reference_point_repeats,
                        select_last=False,
                        fixed_scale=None,
                        update_ratio=True,
                        max_steps_limit=reference_max_steps_limit,
                        reference_token_count_mode="actual",
                        diverse_queries_per_round=True,
                    )
                    if constraint_target == "tps":
                        base_constraint_tps = max(
                            1e-9,
                            float(center_metrics.get("token_per_second", runner.reference_tps)),
                        )
                        auto_reference_constraint_center_tps = float(base_constraint_tps)
                        print(
                            "[Reference] Auto TPS constraint center from blend(cs=0.5): "
                            f"{base_constraint_tps:.6f} tok/s"
                        )
                    else:
                        center_metric_per_token = float(
                            center_metrics.get(
                                "objective_per_token",
                                center_metrics.get("cost_per_token", 0.0),
                            )
                        )
                        base_constraint_1m = max(1e-9, center_metric_per_token * 1_000_000.0)
                        auto_reference_constraint_center_per_1m = float(base_constraint_1m)
                        if objective_metric in cost_objective_metrics:
                            print(
                                "[Reference] Auto constraint center from blend(cs=0.5): "
                                f"${base_constraint_1m:.6f}/1M"
                            )
                        else:
                            print(
                                "[Reference] Auto constraint center from blend(cs=0.5): "
                                f"{base_constraint_1m:.6f} kWh/1M"
                            )
                finally:
                    runner.objective_selection_mode = saved_mode
                    runner.constraint_target = saved_constraint_target
                    runner.metric_constraint_per_token = saved_constraint
                    runner.min_tps_constraint = saved_min_tps

                def _measure_constraint_curve(
                    center_value: float,
                    repeats: int,
                    select_last: bool,
                    fixed_scale: Optional[float],
                    update_ratio: bool,
                    max_steps_limit: Optional[int],
                ) -> List[dict]:
                    anchors = sorted(
                        {
                            max(1e-9, float(center_value) * float(m))
                            for m in reference_constraint_multipliers_list
                        }
                    )
                    rows = []
                    original_constraint = runner.metric_constraint_per_token
                    original_min_tps = runner.min_tps_constraint
                    original_mode = runner.objective_selection_mode
                    original_target = getattr(runner, "constraint_target", "metric")
                    try:
                        runner.objective_selection_mode = "constraint"
                        runner.constraint_target = constraint_target
                        for anchor in anchors:
                            if constraint_target == "tps":
                                runner.metric_constraint_per_token = None
                                runner.min_tps_constraint = float(anchor)
                            else:
                                runner.metric_constraint_per_token = float(anchor) / 1_000_000.0
                                runner.min_tps_constraint = None
                            anchor_metrics = _run_reference_warmup(
                                cost_sensitivity_value=0.5,
                                repeats=repeats,
                                select_last=select_last,
                                fixed_scale=fixed_scale,
                                update_ratio=update_ratio,
                                max_steps_limit=max_steps_limit,
                                reference_token_count_mode=(
                                    "clipped_expected"
                                    if bool(select_last) and fixed_scale is not None
                                    else "actual"
                                ),
                                diverse_queries_per_round=True,
                                aggregate_query_means=(not bool(select_last)),
                            )
                            row = _build_anchor_row(
                                anchor_value=0.5,
                                anchor_metrics=anchor_metrics,
                                fixed_scale=fixed_scale,
                            )
                            row.pop("cost_sensitivity", None)
                            if constraint_target == "tps":
                                row["min_tps_constraint"] = float(anchor)
                            else:
                                row["metric_constraint_per_1m_token"] = float(anchor)
                            rows.append(row)
                    finally:
                        runner.metric_constraint_per_token = original_constraint
                        runner.min_tps_constraint = original_min_tps
                        runner.objective_selection_mode = original_mode
                        runner.constraint_target = original_target
                    return rows

                center_value = (
                    float(base_constraint_tps)
                    if constraint_target == "tps"
                    else float(base_constraint_1m)
                )
                selector_key = "min_tps_constraint" if constraint_target == "tps" else "metric_constraint_per_1m_token"
                pass1_curve = _measure_constraint_curve(
                    center_value=center_value,
                    repeats=reference_point_repeats,
                    select_last=False,
                    fixed_scale=None,
                    update_ratio=True,
                    max_steps_limit=reference_max_steps_limit,
                )
                pass1_ref_row = (
                    min(
                        pass1_curve,
                        key=lambda row: abs(
                            float(row.get(selector_key, 0.0)) - float(center_value)
                        ),
                    )
                    if pass1_curve else None
                )
                recentered_constraint_value = center_value
                if pass1_ref_row is not None:
                    if constraint_target == "tps":
                        measured_center_value = float(pass1_ref_row.get("predicted_tps", 0.0))
                    else:
                        measured_center_value = float(pass1_ref_row.get("predicted_metric_per_1m_token", 0.0))
                    if np.isfinite(measured_center_value) and measured_center_value > 0:
                        recentered_constraint_value = max(1e-9, float(measured_center_value))

                if abs(recentered_constraint_value - float(center_value)) / max(1e-9, float(center_value)) > 1e-3:
                    if constraint_target == "tps":
                        print(
                            "[Reference] Re-centering TPS constraint anchors: "
                            f"{center_value:.6f} tok/s -> {recentered_constraint_value:.6f} tok/s"
                        )
                    elif objective_metric in cost_objective_metrics:
                        print(
                            "[Reference] Re-centering constraint anchors: "
                            f"${center_value:.6f}/1M -> ${recentered_constraint_value:.6f}/1M"
                        )
                    else:
                        print(
                            "[Reference] Re-centering constraint anchors: "
                            f"{center_value:.6f} -> {recentered_constraint_value:.6f} kWh/1M"
                        )
                # 2) recentered anchors calibration: dynamic scale
                _measure_constraint_curve(
                    center_value=recentered_constraint_value,
                    repeats=reference_point_repeats,
                    select_last=False,
                    fixed_scale=None,
                    update_ratio=True,
                    max_steps_limit=reference_max_steps_limit,
                )
                fixed_reference_accept_scale = (
                    float(runner.get_accept_length_ratio_mean())
                    if runner.get_accept_length_ratio_mean() is not None
                    else 1.0
                )
                print(
                    "[Reference] Fixed accept_length_scale after calibration: "
                    f"{fixed_reference_accept_scale:.6f}"
                )
                # 3) final evaluation: scale
                reference_constraint_anchor_curve = _measure_constraint_curve(
                    center_value=recentered_constraint_value,
                    repeats=reference_point_repeats,
                    select_last=False,
                    fixed_scale=fixed_reference_accept_scale,
                    update_ratio=False,
                    max_steps_limit=reference_max_steps_limit,
                )
                if constraint_target == "tps":
                    auto_reference_constraint_center_tps = float(recentered_constraint_value)
                else:
                    auto_reference_constraint_center_per_1m = float(recentered_constraint_value)

                if reference_constraint_anchor_curve:
                    ref_row = min(
                        reference_constraint_anchor_curve,
                        key=lambda row: abs(
                            float(row.get(selector_key, 0.0)) - float(recentered_constraint_value)
                        ),
                    )
                else:
                    ref_row = None
                reference_tps = float(ref_row.get("predicted_tps", 0.0)) if ref_row else 0.0
                reference_cost_per_token = float(ref_row.get("predicted_cost_per_token", 0.0)) if ref_row else 0.0
                reference_obj_per_token = float(ref_row.get("predicted_objective_per_token", 0.0)) if ref_row else 0.0
                reference_draft_objective_rate_per_sec = float(
                    ref_row.get("draft_objective_rate_per_sec", 0.0)
                ) if ref_row else 0.0
                reference_target_objective_rate_per_sec = float(
                    ref_row.get("target_objective_rate_per_sec", 0.0)
                ) if ref_row else 0.0
                if reference_tps > 0:
                    runner.reference_tps = float(reference_tps)
                if reference_cost_per_token > 0:
                    runner.reference_cost_per_token = float(reference_cost_per_token)
                if reference_obj_per_token > 0:
                    runner.reference_objective_per_token = float(reference_obj_per_token)
                if runner.uses_draft_energy_objective() and reference_draft_objective_rate_per_sec > 0:
                    runner.draft_objective_rate_per_sec = float(reference_draft_objective_rate_per_sec)
                if runner.uses_target_energy_objective() and reference_target_objective_rate_per_sec > 0:
                    runner.target_objective_rate_per_sec = float(reference_target_objective_rate_per_sec)

                metric_vals = [
                    float(row.get("predicted_metric_per_token", 0.0))
                    for row in reference_constraint_anchor_curve
                ]
                tps_vals = [
                    float(row.get("predicted_tps", 0.0))
                    for row in reference_constraint_anchor_curve
                ]
                if metric_vals and tps_vals:
                    reference_feasible_metric_per_token = {
                        "min": float(min(metric_vals)),
                        "max": float(max(metric_vals)),
                        "mean": float(sum(metric_vals) / len(metric_vals)),
                        "count": int(len(metric_vals)),
                    }
                    reference_feasible_tps = {
                        "min": float(min(tps_vals)),
                        "max": float(max(tps_vals)),
                        "mean": float(sum(tps_vals) / len(tps_vals)),
                        "count": int(len(tps_vals)),
                    }
                else:
                    reference_feasible_metric_per_token = None
                    reference_feasible_tps = None
                reference_tradeoff_curve_by_constraint = _build_reference_tradeoff_curve_by_constraint(
                    reference_constraint_anchor_curve,
                    constraint_target=constraint_target,
                )
                reference_cs_anchor_curve = None
                reference_tradeoff_curve_cs0_1 = None
            else:
                print("[Reference] No cache found. Running full-query cs anchors independently.")
                cs_anchors = [0.0, 0.5, 1.0]
                # 1) Calibration over cs anchors to estimate actual/expected scale.
                for cs_anchor in cs_anchors:
                    _run_reference_warmup(
                        cost_sensitivity_value=float(cs_anchor),
                        repeats=reference_point_repeats,
                        select_last=False,
                        fixed_scale=None,
                        update_ratio=True,
                        max_steps_limit=reference_max_steps_limit,
                        reference_token_count_mode="actual",
                        diverse_queries_per_round=True,
                    )
                fixed_reference_accept_scale = (
                    float(runner.get_accept_length_ratio_mean())
                    if runner.get_accept_length_ratio_mean() is not None
                    else 1.0
                )
                print(
                    "[Reference] Fixed accept_length_scale after calibration: "
                    f"{fixed_reference_accept_scale:.6f}"
                )

                # 2) Final evaluation with the fixed scale.
                reference_cs_anchor_curve = []
                anchor_metrics_map = {}
                for cs_anchor in cs_anchors:
                    anchor_metrics = _run_reference_warmup(
                        cost_sensitivity_value=float(cs_anchor),
                        repeats=reference_point_repeats,
                        select_last=False,
                        fixed_scale=fixed_reference_accept_scale,
                        update_ratio=False,
                        max_steps_limit=reference_max_steps_limit,
                        reference_token_count_mode="clipped_expected",
                        diverse_queries_per_round=True,
                        aggregate_query_means=True,
                    )
                    row = _build_anchor_row(
                        anchor_value=float(cs_anchor),
                        anchor_metrics=anchor_metrics,
                        fixed_scale=fixed_reference_accept_scale,
                    )
                    reference_cs_anchor_curve.append(row)
                    anchor_metrics_map[round(float(cs_anchor), 6)] = row

                # Select the reference row for the requested cs.
                ref_row = _select_reference_anchor_for_cs(
                    reference_cs_anchor_curve,
                    float(runner.cost_sensitivity),
                )
                if ref_row is None:
                    ref_row = anchor_metrics_map.get(0.5, reference_cs_anchor_curve[0] if reference_cs_anchor_curve else None)
                reference_tps = float(ref_row.get("predicted_tps", 0.0)) if ref_row else 0.0
                reference_cost_per_token = float(ref_row.get("predicted_cost_per_token", 0.0)) if ref_row else 0.0
                reference_obj_per_token = float(ref_row.get("predicted_objective_per_token", 0.0)) if ref_row else 0.0
                reference_draft_objective_rate_per_sec = float(ref_row.get("draft_objective_rate_per_sec", 0.0)) if ref_row else 0.0
                reference_target_objective_rate_per_sec = float(ref_row.get("target_objective_rate_per_sec", 0.0)) if ref_row else 0.0
                if isinstance(ref_row, dict):
                    if bool(ref_row.get("_selected_ref_interpolated", False)):
                        print(
                            "[Reference] CS-specific reference selected "
                            f"(cs={float(runner.cost_sensitivity):.2g}, "
                            f"interp={float(ref_row.get('_selected_ref_cs_left', 0.0)):.2g}"
                            f"~{float(ref_row.get('_selected_ref_cs_right', 0.0)):.2g})."
                        )
                    else:
                        print(
                            "[Reference] CS-specific reference selected "
                            f"(cs={float(ref_row.get('_selected_ref_cs', runner.cost_sensitivity)):.2g})."
                        )
                if reference_tps > 0:
                    runner.reference_tps = float(reference_tps)
                if reference_cost_per_token > 0:
                    runner.reference_cost_per_token = float(reference_cost_per_token)
                if reference_obj_per_token > 0:
                    runner.reference_objective_per_token = float(reference_obj_per_token)
                if runner.uses_draft_energy_objective() and reference_draft_objective_rate_per_sec > 0:
                    runner.draft_objective_rate_per_sec = float(reference_draft_objective_rate_per_sec)
                if runner.uses_target_energy_objective() and reference_target_objective_rate_per_sec > 0:
                    runner.target_objective_rate_per_sec = float(reference_target_objective_rate_per_sec)
                # Feasible range endpoints: cs=0 and cs=1.
                row0 = anchor_metrics_map.get(0)
                row1 = anchor_metrics_map.get(1.0)
                if row0 is not None and row1 is not None:
                    m0 = float(row0.get("predicted_metric_per_token", 0.0))
                    m1 = float(row1.get("predicted_metric_per_token", 0.0))
                    t0 = float(row0.get("predicted_tps", 0.0))
                    t1 = float(row1.get("predicted_tps", 0.0))
                    reference_feasible_metric_per_token = {
                        "min": float(min(m0, m1)),
                        "max": float(max(m0, m1)),
                        "mean": float((m0 + m1) / 2.0),
                        "count": 2,
                    }
                    reference_feasible_tps = {
                        "min": float(min(t0, t1)),
                        "max": float(max(t0, t1)),
                        "mean": float((t0 + t1) / 2.0),
                        "count": 2,
                    }
                else:
                    reference_feasible_metric_per_token = None
                    reference_feasible_tps = None

                reference_tradeoff_curve_cs0_1 = _build_reference_tradeoff_curve(
                    reference_cs_anchor_curve,
                    step=0.05,
                )
                reference_constraint_anchor_curve = None
                reference_tradeoff_curve_by_constraint = None

            reference_metrics = {
                "token_per_second": float(runner.reference_tps),
                "cost_per_token": float(runner.reference_cost_per_token),
                "objective_per_token": float(runner.reference_objective_per_token),
                "constraint_target": str(constraint_target),
                "draft_objective_rate_per_sec": float(reference_draft_objective_rate_per_sec),
                "target_objective_rate_per_sec": float(reference_target_objective_rate_per_sec),
                "feasible_metric_per_token": reference_feasible_metric_per_token,
                "feasible_tps": reference_feasible_tps,
                "reference_cs_anchor_curve": reference_cs_anchor_curve,
                "reference_tradeoff_curve_cs0_1": reference_tradeoff_curve_cs0_1,
                "reference_constraint_anchor_curve": reference_constraint_anchor_curve,
                "reference_tradeoff_curve_by_constraint": reference_tradeoff_curve_by_constraint,
                "reference_constraint_center_per_1m_token": (
                    float(auto_reference_constraint_center_per_1m)
                    if auto_reference_constraint_center_per_1m is not None
                    else None
                ),
                "reference_point_repeat_count": int(reference_point_repeats),
                "reference_max_steps_limit": int(reference_max_steps_limit),
                "reference_point_selection_rule": "mean_over_queries_with_fixed_scale_after_calibration",
                "reference_fixed_accept_length_scale": (
                    float(fixed_reference_accept_scale)
                    if fixed_reference_accept_scale is not None
                    else None
                ),
                "reference_accept_length_last_trial": (
                    {
                        "actual_accept_length_last": ref_row.get("actual_accept_length_last", None),
                        "expected_accept_length_raw_last": ref_row.get("expected_accept_length_raw_last", None),
                        "expected_accept_length_scaled_last": ref_row.get("expected_accept_length_scaled_last", None),
                        "expected_accept_length_clipped_last": ref_row.get("expected_accept_length_clipped_last", None),
                        "accept_length_error_last": ref_row.get("accept_length_error_last", None),
                        "accept_length_error_ratio_last": ref_row.get("accept_length_error_ratio_last", None),
                    }
                    if ref_row is not None
                    else None
                ),
                "reference_monotonicity_summary": (
                    _analyze_reference_tradeoff_curve(reference_tradeoff_curve_cs0_1)
                    if objective_selection_mode != "constraint"
                    else None
                ),
                "reference_cause_summary": (
                    _build_reference_cause_summary(
                        reference_cs_anchor_curve,
                        reference_tradeoff_curve_cs0_1,
                    )
                    if objective_selection_mode != "constraint"
                    else None
                ),
                "reference_debug_trace": (
                    {"anchor_rows": reference_cs_anchor_curve}
                    if isinstance(reference_cs_anchor_curve, list)
                    else None
                ),
            }
            save_reference_cache(
                cache_path=reference_cache_path,
                warmup_metrics=reference_metrics,
                objective_selection_mode=objective_selection_mode,
            )
            print(f"[Reference] Saved reference cache: {reference_cache_path}")
        if objective_selection_mode == "constraint":
            if (
                (not isinstance(reference_tradeoff_curve_by_constraint, list) or not reference_tradeoff_curve_by_constraint)
                and isinstance(reference_constraint_anchor_curve, list)
            ):
                reference_tradeoff_curve_by_constraint = _build_reference_tradeoff_curve_by_constraint(
                    reference_constraint_anchor_curve,
                    constraint_target=constraint_target,
                )
        else:
            if (
                (not isinstance(reference_tradeoff_curve_cs0_1, list) or not reference_tradeoff_curve_cs0_1)
                and isinstance(reference_cs_anchor_curve, list)
            ):
                reference_tradeoff_curve_cs0_1 = _build_reference_tradeoff_curve(
                    reference_cs_anchor_curve,
                    step=0.05,
                )
        if objective_selection_mode == "constraint":
            runner.constraint_target = constraint_target
            if constraint_target == "tps":
                runner.metric_constraint_per_token = None
                if min_tps_constraint is None or float(min_tps_constraint) <= 0:
                    inferred_center_tps = (
                        float(auto_reference_constraint_center_tps)
                        if auto_reference_constraint_center_tps is not None
                        else float(runner.reference_tps)
                    )
                    min_tps_constraint = max(1e-9, float(inferred_center_tps))
                    runner.min_tps_constraint = float(min_tps_constraint)
                    print(
                        "[INFO] --min-tps-constraint not provided; "
                        f"using auto center {float(min_tps_constraint):.6f} tok/s."
                    )
                else:
                    runner.min_tps_constraint = float(min_tps_constraint)
            else:
                runner.min_tps_constraint = None
                if metric_constraint_per_1m_token is None:
                    inferred_center_1m = (
                        float(auto_reference_constraint_center_per_1m)
                        if auto_reference_constraint_center_per_1m is not None
                        else float(runner.reference_objective_per_token) * 1_000_000.0
                    )
                    metric_constraint_per_1m_token = max(1e-9, float(inferred_center_1m))
                    metric_constraint_per_token = float(metric_constraint_per_1m_token) / 1_000_000.0
                    runner.metric_constraint_per_token = float(metric_constraint_per_token)
                    if objective_metric in cost_objective_metrics:
                        print(
                            "[INFO] --metric-constraint-per-1m-token not provided; "
                            f"using auto center ${float(metric_constraint_per_1m_token):.6f}/1M."
                        )
                    else:
                        print(
                            "[INFO] --metric-constraint-per-1m-token not provided; "
                            f"using auto center {float(metric_constraint_per_1m_token):.6f} kWh/1M."
                        )
                else:
                    metric_constraint_per_token = float(metric_constraint_per_1m_token) / 1_000_000.0
                    runner.metric_constraint_per_token = float(metric_constraint_per_token)
        print(
            "[Warmup Reference] "
            f"reference_tps={runner.reference_tps:.6f}, "
            f"reference_cost_per_token={runner.reference_cost_per_token:.12f}, "
            f"reference_objective_per_token={runner.reference_objective_per_token:.12f}"
        )
        if isinstance(reference_feasible_metric_per_token, dict):
            f_min = reference_feasible_metric_per_token.get("min", None)
            f_max = reference_feasible_metric_per_token.get("max", None)
            if f_min is not None and f_max is not None:
                if objective_metric in cost_objective_metrics:
                    print(
                        "[Warmup Reference] feasible metric range (approx): "
                        f"${float(f_min)*1_000_000.0:.6f} ~ ${float(f_max)*1_000_000.0:.6f} per 1M tokens"
                    )
                else:
                    print(
                        "[Warmup Reference] feasible metric range (approx): "
                        f"{float(f_min)*1_000_000.0:.6f} ~ {float(f_max)*1_000_000.0:.6f} kWh per 1M tokens"
                    )
                if (
                    objective_selection_mode == "constraint"
                    and constraint_target == "metric"
                    and metric_constraint_per_token is not None
                    and metric_constraint_per_token > 0
                ):
                    c = float(metric_constraint_per_token)
                    f_min_val = float(f_min)
                    f_max_val = float(f_max)
                    if c < f_min_val:
                        unit = "$" if objective_metric in cost_objective_metrics else "kWh"
                        print(
                            "[WARN] metric constraint is tighter than reference feasible lower bound. "
                            f"constraint={unit}{c*1_000_000.0:.6f}/1M, "
                            f"feasible_min={unit}{f_min_val*1_000_000.0:.6f}/1M. "
                            "Most steps may be infeasible; fallback behavior may dominate."
                        )
                    elif c > f_max_val:
                        unit = "$" if objective_metric in cost_objective_metrics else "kWh"
                        print(
                            "[WARN] metric constraint is looser than reference feasible upper bound. "
                            f"constraint={unit}{c*1_000_000.0:.6f}/1M, "
                            f"feasible_max={unit}{f_max_val*1_000_000.0:.6f}/1M. "
                            "Constraint may have little practical effect."
                        )
        if isinstance(reference_feasible_tps, dict):
            t_min = reference_feasible_tps.get("min", None)
            t_max = reference_feasible_tps.get("max", None)
            if t_min is not None and t_max is not None:
                print(
                    "[Warmup Reference] feasible hybrid TPS range (approx): "
                    f"{float(t_min):.6f} ~ {float(t_max):.6f}"
                )
                if (
                    objective_selection_mode == "constraint"
                    and constraint_target == "tps"
                    and min_tps_constraint is not None
                    and float(min_tps_constraint) > 0
                ):
                    c = float(min_tps_constraint)
                    t_min_val = float(t_min)
                    t_max_val = float(t_max)
                    if c > t_max_val:
                        print(
                            "[WARN] TPS constraint is higher than reference feasible upper bound. "
                            f"constraint={c:.6f} tok/s, feasible_max={t_max_val:.6f} tok/s. "
                            "Most steps may be infeasible; fallback behavior may dominate."
                        )
                    elif c < t_min_val:
                        print(
                            "[WARN] TPS constraint is lower than reference feasible lower bound. "
                            f"constraint={c:.6f} tok/s, feasible_min={t_min_val:.6f} tok/s. "
                            "Constraint may have little practical effect."
                        )
        if objective_selection_mode == "constraint":
            if isinstance(reference_constraint_anchor_curve, list) and reference_constraint_anchor_curve:
                print("[Warmup Reference] constraint anchor curve (predicted):")
                for row in reference_constraint_anchor_curve:
                    tps = float(row.get("predicted_tps", 0.0))
                    obj_1m = float(
                        row.get(
                            "predicted_metric_per_1m_token",
                            row.get("predicted_objective_per_1m_token", 0.0),
                        )
                    )
                    if constraint_target == "tps":
                        c_tps = float(row.get("min_tps_constraint", 0.0))
                        if objective_metric in cost_objective_metrics:
                            print(f"  - min_tps={c_tps:.6f}: tps={tps:.6f}, cost/1M=${obj_1m:.6f}")
                        else:
                            print(f"  - min_tps={c_tps:.6f}: tps={tps:.6f}, energy/1M={obj_1m:.6f} kWh")
                    elif objective_metric in cost_objective_metrics:
                        c1m = float(row.get("metric_constraint_per_1m_token", 0.0))
                        print(f"  - constraint=${c1m:.6f}/1M: tps={tps:.6f}, cost/1M=${obj_1m:.6f}")
                    else:
                        c1m = float(row.get("metric_constraint_per_1m_token", 0.0))
                        print(f"  - constraint={c1m:.6f} kWh/1M: tps={tps:.6f}, energy/1M={obj_1m:.6f} kWh")
            if isinstance(reference_tradeoff_curve_by_constraint, list) and reference_tradeoff_curve_by_constraint:
                print(
                    "[Warmup Reference] trade-off curve available for constraint anchors "
                    f"({len(reference_tradeoff_curve_by_constraint)} points)."
                )
        else:
            if isinstance(reference_cs_anchor_curve, list) and reference_cs_anchor_curve:
                print("[Warmup Reference] cs anchor curve (predicted):")
                for row in reference_cs_anchor_curve:
                    cs = float(row.get("cost_sensitivity", 0.0))
                    tps = float(row.get("predicted_tps", 0.0))
                    obj_1m = float(
                        row.get(
                            "predicted_metric_per_1m_token",
                            row.get("predicted_objective_per_1m_token", 0.0),
                        )
                    )
                    if objective_metric in cost_objective_metrics:
                        print(f"  - cs={cs:.2g}: tps={tps:.6f}, cost/1M=${obj_1m:.6f}")
                    else:
                        print(f"  - cs={cs:.2g}: tps={tps:.6f}, energy/1M={obj_1m:.6f} kWh")
            if isinstance(reference_tradeoff_curve_cs0_1, list) and reference_tradeoff_curve_cs0_1:
                print(
                    "[Warmup Reference] trade-off curve available for cs=0~1 "
                    f"({len(reference_tradeoff_curve_cs0_1)} points, step=0.05)."
                )
        if not isinstance(reference_feasible_metric_per_token, dict):
            # Fallback to a single-point feasible range.
            point_metric = (
                float(runner.reference_cost_per_token)
                if objective_metric == "total_cost"
                else float(runner.reference_objective_per_token)
            )
            reference_feasible_metric_per_token = {
                "min": point_metric,
                "max": point_metric,
                "mean": point_metric,
                "count": 1,
            }
            print(
                "[WARN] reference feasible metric range missing in cache; "
                "using reference point as fallback range."
            )
        if not isinstance(reference_feasible_tps, dict):
            point_tps = float(runner.reference_tps)
            reference_feasible_tps = {
                "min": point_tps,
                "max": point_tps,
                "mean": point_tps,
                "count": 1,
            }
            print(
                "[WARN] reference feasible TPS range missing in cache; "
                "using reference TPS as fallback range."
            )
        reset_gpu_monitor_after_reference(runner)
        server_only_mode = False
        server_only_tps = None
        if force_server_only_ar:
            server_only_mode = True
            print("[INFO] server-only AR force enabled by --force-server-only-ar.")
        elif force_server_only:
            server_only_mode = True
            print("[INFO] server-only force enabled by --force-server-only.")
        if not disable_server_only:
            server_only_tps = load_server_only_baseline_metric(
                baseline_json=server_only_baseline_json,
                base_model_path=base_model_path,
                draft_model_path=draft_model_path,
            )
        else:
            print("[INFO] server-only disabled by --disable-server-only.")
        if server_only_tps is not None and server_only_tps > 0:
            if objective_metric in cost_objective_metrics:
                so_metric_per_token = (target_per_sec_cost / server_only_tps) if server_only_tps > 0 else float("inf")
                print(
                    "[Warmup Reference] predicted server-only range (approx): "
                    f"metric=${so_metric_per_token*1_000_000.0:.6f}/1M, "
                    f"tps={float(server_only_tps):.6f}"
                )
            else:
                print(
                    "[Warmup Reference] predicted server-only range (approx): "
                    f"metric={0.0:.6f} kWh/1M (draft-only objective), "
                    f"tps={float(server_only_tps):.6f}"
                )
            alpha = runner.get_sensitivity_alpha()
            ref_tps = max(1e-9, runner.reference_tps)
            ref_obj = runner.get_reference_objective_per_token()
            warmup_score = 1.0  # reference
            if objective_metric in cost_objective_metrics:
                server_only_obj_per_token = (target_per_sec_cost / server_only_tps) if server_only_tps > 0 else float("inf")
            else:
                # energy : server-only draft objective 0
                server_only_obj_per_token = 0.0
            server_only_score = (
                alpha * (server_only_obj_per_token / ref_obj)
                + (1.0 - alpha) * (ref_tps / max(1e-9, server_only_tps))
            )
            if force_server_only or server_only_score < warmup_score:
                server_only_mode = True
                print("Entering server-only mode...")
            else:
                print("Staying in draft-target mode (server-only not beneficial by normalized objective).")
        if (
            server_only_mode
            and objective_metric in {"total_cost", "api_cost"}
            and (not bool(no_draft_cost))
            and (not bool(getattr(runner, "bill_draft_as_target_gpu", False)))
        ):
            print(
                "[INFO] server-only cost objective -> using time-based draft billing "
                "(draft+target time billed with target_per_sec_cost)."
            )
            runner.bill_draft_as_target_gpu = True
            bill_draft_as_target_gpu = True
            draft_per_sec_cost = float(target_per_sec_cost)
            runner.draft_per_sec_cost = float(target_per_sec_cost)
            runner.draft_objective_rate_per_sec = float(target_per_sec_cost)
        runner.server_only_mode = bool(server_only_mode)

        if reference_test_mode:
            probe_runs = max(1, int(reference_test_runs))
            hybrid_metric_obs_per_1m = []
            hybrid_tps_obs = []
            server_only_metric_obs_per_1m = []
            server_only_tps_obs = []

            for _ in range(probe_runs):
                hybrid_probe = run_warmup(
                    sock=sock,
                    runner=runner,
                    tokenizer=tokenizer,
                    questions=questions,
                    base_model_path=base_model_path,
                    bench_name=bench_name,
                    nodes=nodes,
                    max_depth=max_depth,
                    per_token_probability_bound=per_token_probability_bound,
                    per_path_probability_bound=per_path_probability_bound,
                    min_width=min_width,
                    fixed_width=fixed_width,
                    fixed_width_value=fixed_width_value,
                    fixed_nnodes=fixed_nnodes,
                    fixed_depth=fixed_depth,
                    debug=debug,
                    warmup_cost_sensitivity=float(cost_sensitivity),
                    warmup_rounds=1,
                    full_query=False,
                )
                hybrid_tps_obs.append(float(hybrid_probe.get("token_per_second", 0.0)))
                if objective_metric == "total_cost":
                    hybrid_metric_obs_per_1m.append(float(hybrid_probe.get("cost_per_token", 0.0)) * 1_000_000.0)
                else:
                    hybrid_metric_obs_per_1m.append(float(hybrid_probe.get("objective_per_token", 0.0)) * 1_000_000.0)

                if server_only_tps is not None and server_only_tps > 0:
                    so_probe = run_server_only_probe(
                        sock=sock,
                        questions=questions,
                        tokenizer=tokenizer,
                        base_model_path=base_model_path,
                        bench_name=bench_name,
                        nodes=nodes,
                        max_depth=max_depth,
                        per_token_probability_bound=per_token_probability_bound,
                        per_path_probability_bound=per_path_probability_bound,
                        min_width=min_width,
                        fixed_width=fixed_width,
                        fixed_width_value=fixed_width_value,
                        fixed_nnodes=fixed_nnodes,
                        fixed_depth=fixed_depth,
                        proactive_drafting=proactive_drafting,
                        proactive_threshold=proactive_threshold,
                        adaptive_proactive_threshold=adaptive_proactive_threshold,
                        cost_sensitivity=cost_sensitivity,
                        draft_per_sec_cost=draft_per_sec_cost,
                        target_per_sec_cost=target_per_sec_cost,
                        reference_tps=runner.reference_tps,
                        reference_objective_per_token=runner.reference_objective_per_token,
                        objective_metric=objective_metric,
                        no_draft_cost=no_draft_cost,
                        bill_draft_as_target_gpu=bool(getattr(runner, "bill_draft_as_target_gpu", False)),
                        server_draft_profile_auto=bool(server_draft_profile_auto),
                        server_draft_profile_force_refresh=bool(server_draft_profile_force_refresh),
                        server_draft_profile_model_calls_per_count=int(max(1, int(server_draft_profile_model_calls_per_count))),
                        server_draft_profile_width_list=str(server_draft_profile_width_list),
                        question_file=question_file,
                        server_name=server_name,
                    )
                    server_only_tps_obs.append(float(so_probe.get("token_per_second", 0.0)))
                    server_only_metric_obs_per_1m.append(float(so_probe.get("metric_per_token", 0.0)) * 1_000_000.0)

            def _mean(xs):
                return float(sum(xs) / len(xs)) if xs else None

            def _range_dict(xs):
                return {
                    "min": float(min(xs)) if xs else None,
                    "max": float(max(xs)) if xs else None,
                    "mean": _mean(xs),
                    "count": int(len(xs)),
                }

            def _coverage(xs, lo, hi):
                if not xs or lo is None or hi is None:
                    return None
                hit = sum(1 for x in xs if float(lo) <= float(x) <= float(hi))
                return float(hit) / float(len(xs))

            hybrid_pred_metric = None
            hybrid_pred_tps = None
            if isinstance(reference_feasible_metric_per_token, dict):
                if reference_feasible_metric_per_token.get("min") is not None and reference_feasible_metric_per_token.get("max") is not None:
                    hybrid_pred_metric = {
                        "min": float(reference_feasible_metric_per_token["min"]) * 1_000_000.0,
                        "max": float(reference_feasible_metric_per_token["max"]) * 1_000_000.0,
                    }
            if isinstance(reference_feasible_tps, dict):
                if reference_feasible_tps.get("min") is not None and reference_feasible_tps.get("max") is not None:
                    hybrid_pred_tps = {
                        "min": float(reference_feasible_tps["min"]),
                        "max": float(reference_feasible_tps["max"]),
                    }

            server_only_pred_metric = None
            server_only_pred_tps = None
            if server_only_tps is not None and server_only_tps > 0:
                if objective_metric in cost_objective_metrics:
                    so_metric = (float(target_per_sec_cost) / float(server_only_tps)) * 1_000_000.0
                else:
                    so_metric = 0.0
                server_only_pred_metric = {"min": float(so_metric), "max": float(so_metric)}
                server_only_pred_tps = {"min": float(server_only_tps), "max": float(server_only_tps)}

            report = {
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "objective_metric": str(objective_metric),
                "objective_selection_mode": str(objective_selection_mode),
                "reference_test_runs": int(probe_runs),
                "hybrid": {
                    "predicted_metric_per_1m_range": hybrid_pred_metric,
                    "predicted_tps_range": hybrid_pred_tps,
                    "observed_metric_per_1m": _range_dict(hybrid_metric_obs_per_1m),
                    "observed_tps": _range_dict(hybrid_tps_obs),
                    "metric_range_coverage": _coverage(
                        hybrid_metric_obs_per_1m,
                        None if hybrid_pred_metric is None else hybrid_pred_metric["min"],
                        None if hybrid_pred_metric is None else hybrid_pred_metric["max"],
                    ),
                    "tps_range_coverage": _coverage(
                        hybrid_tps_obs,
                        None if hybrid_pred_tps is None else hybrid_pred_tps["min"],
                        None if hybrid_pred_tps is None else hybrid_pred_tps["max"],
                    ),
                },
                "server_only": {
                    "predicted_metric_per_1m_range": server_only_pred_metric,
                    "predicted_tps_range": server_only_pred_tps,
                    "observed_metric_per_1m": _range_dict(server_only_metric_obs_per_1m),
                    "observed_tps": _range_dict(server_only_tps_obs),
                    "metric_range_coverage": _coverage(
                        server_only_metric_obs_per_1m,
                        None if server_only_pred_metric is None else server_only_pred_metric["min"],
                        None if server_only_pred_metric is None else server_only_pred_metric["max"],
                    ),
                    "tps_range_coverage": _coverage(
                        server_only_tps_obs,
                        None if server_only_pred_tps is None else server_only_pred_tps["min"],
                        None if server_only_pred_tps is None else server_only_pred_tps["max"],
                    ),
                },
            }

            if reference_test_output_json:
                report_path = reference_test_output_json
            else:
                script_dir = os.path.dirname(__file__)
                parent_dir = os.path.dirname(script_dir)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_path = os.path.join(parent_dir, "result", f"reference_range_test_{objective_metric}_{ts}.json")
            try:
                with open(report_path, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                print(f"[Reference Test] saved: {report_path}")
            except Exception as e:
                print(f"[WARN] failed to write reference test report ({report_path}): {e}")

            print(
                "[Reference Test] hybrid coverage: "
                f"metric={report['hybrid']['metric_range_coverage']}, "
                f"tps={report['hybrid']['tps_range_coverage']}"
            )
            print(
                "[Reference Test] server-only coverage: "
                f"metric={report['server_only']['metric_range_coverage']}, "
                f"tps={report['server_only']['tps_range_coverage']}"
            )
            return

        if bool(reference_only_exit_after_cache):
            print("[ReferenceOnly] Reference cache ready; skipping answer decoding loop.")
            return {
                "mode": "reference_only",
                "reference_cache_path": str(reference_cache_path),
                "reference_tps": float(runner.reference_tps),
                "reference_cost_per_token": float(runner.reference_cost_per_token),
                "reference_objective_per_token": float(runner.reference_objective_per_token),
                "objective_metric": str(objective_metric),
                "cost_sensitivity": float(cost_sensitivity),
            }

        grand_total_steps = 0
        grand_total_accepted = 0
        grand_total_draft_tokens = 0
        grand_total_new_tokens = 0
        # Query reference .
        # - query /
        # - speculative decoding step (tree_build + network + target verify)
        ref_update_total_time_sec = 0.0
        ref_update_total_tokens = 0.0
        ref_update_total_metric = 0.0
        proactive_draft_attempts = 0
        proactive_tree_used = 0
        proactive_used_path_prob_sum = 0.0
        proactive_used_path_prob_count = 0
        proactive_unused_path_prob_sum = 0.0
        proactive_unused_path_prob_count = 0
        proactive_used_tree_steps = 0
        proactive_used_tree_depth_sum = 0.0
        proactive_used_tree_width_sum = 0.0
        proactive_used_tree_final_nodes_sum = 0.0
        proactive_used_tree_accept_length_sum = 0.0
        proactive_used_tree_scaled_expected_sum = 0.0
        foreground_tree_steps = 0
        foreground_tree_accept_length_sum = 0.0
        after_proactive_foreground_steps = 0
        after_proactive_foreground_accept_length_sum = 0.0
        after_proactive_cancel_foreground_steps = 0
        after_proactive_cancel_foreground_accept_length_sum = 0.0
        # proactive_total_time_sec/proactive_draft_time_seconds proactive compute .
        # worker thread pause/budget wait wall elapsed .
        # Wall-time gain " tree" reply .
        proactive_total_time_sec = 0.0
        proactive_wall_elapsed_total_sec = 0.0
        proactive_compute_used_sec = 0.0
        proactive_compute_unused_sec = 0.0
        proactive_compute_hidden_used_sec = 0.0
        proactive_compute_after_reply_used_sec = 0.0
        proactive_post_reply_wait_ms_sum = 0.0
        proactive_post_reply_wait_ms_count = 0
        proactive_post_reply_wait_ms_max = 0.0
        proactive_budget_wait_ms_sum = 0.0
        proactive_budget_wait_ms_count = 0
        proactive_budget_wait_ms_max = 0.0
        proactive_resume_after_reply_ms_sum = 0.0
        proactive_resume_after_reply_ms_count = 0
        proactive_resume_after_reply_ms_max = 0.0
        proactive_cancel_to_exit_ms_sum = 0.0
        proactive_cancel_to_exit_ms_count = 0
        proactive_cancel_to_exit_ms_max = 0.0
        proactive_cancel_to_exit_recent_ms = []
        proactive_cancel_pending_count = 0
        proactive_cancel_tree_build_overlap_ms_sum = 0.0
        proactive_cancel_tree_build_overlap_ms_count = 0
        proactive_cancel_tree_build_overlap_ms_max = 0.0
        proactive_cancel_tree_build_overlap_event_count = 0
        proactive_cancel_join_before_tree_build_ms_sum = 0.0
        proactive_cancel_join_before_tree_build_ms_count = 0
        proactive_cancel_join_before_tree_build_ms_max = 0.0
        proactive_cancel_immediate_join_ms_sum = 0.0
        proactive_cancel_immediate_join_ms_count = 0
        proactive_cancel_immediate_join_ms_max = 0.0
        proactive_start_skipped_by_value_count = 0
        proactive_start_expected_gain_ms_sum = 0.0
        proactive_start_expected_cancel_loss_ms_sum = 0.0
        proactive_start_value_score_sum = 0.0
        proactive_start_value_count = 0
        proactive_start_cancel_loss_estimate_ms_latest = None
        proactive_path_match_count = 0
        proactive_path_mismatch_count = 0
        proactive_finalize_early_count = 0
        proactive_expand_continue_count = 0
        proactive_expand_pause_count = 0
        proactive_expected_gain_ms_sum = 0.0
        proactive_expected_loss_ms_sum = 0.0
        proactive_depth_stats = {}
        proactive_expand_depth_counts = {}
        proactive_post_reply_by_depth = {}
        proactive_post_reply_by_width = {}
        proactive_post_reply_by_depth_width = {}
        _pending_canceled_proactive = []
        server_only_total_session_time_sec = 0.0
        server_only_profile_prepare_overhead_sec = 0.0
        last_tree_build_ms = None
        last_target_verification_ms = None
        last_draft_to_target_ms = None
        last_target_to_draft_ms = None
        proactive_budget_ms_latest = None
        run_start_time = time.time()
        metric_spent_total = 0.0
        metric_cap_reached = False
        server_draft_profile_force_refresh_consumed = False

        class _ProactiveTaskHandle:
            """A persistent worker task handle that only provides join/is_alive like Thread."""
            def __init__(self):
                self._done = threading.Event()

            def is_alive(self):
                return not self._done.is_set()

            def join(self, timeout=None):
                self._done.wait(timeout)

            def _mark_done(self):
                self._done.set()

        class _ReusableProactiveWorker:
            """Reuse a single worker thread to avoid creating a new thread for each proactive task."""
            def __init__(self):
                self._queue = queue.Queue()
                self._shutdown = object()
                self.submitted = 0
                self.completed = 0
                self._thread = threading.Thread(target=self._run, daemon=True)
                self._thread.start()

            def submit(self, fn):
                handle = _ProactiveTaskHandle()
                self.submitted += 1
                self._queue.put((fn, handle))
                return handle

            def _run(self):
                while True:
                    item = self._queue.get()
                    if item is self._shutdown:
                        self._queue.task_done()
                        break
                    fn, handle = item
                    try:
                        fn()
                    finally:
                        self.completed += 1
                        handle._mark_done()
                        self._queue.task_done()

            def shutdown(self):
                self._queue.put(self._shutdown)
                self._thread.join()

        proactive_worker = _ReusableProactiveWorker() if proactive_drafting else None
        recv_worker = _ReusableRecvWorker()
        gpu_monitor_long_running = bool(runner.gpu_monitor is not None)
        if gpu_monitor_long_running and not runner.gpu_monitor.monitoring:
            runner.gpu_monitor.start_monitoring()
        if gpu_monitor_long_running and runner.gpu_monitor is not None:
            try:
                runner.gpu_monitor.reset_data()
            except Exception:
                pass

        def _record_cancel_tree_build_overlap(build_start_ts: float, build_end_ts: float) -> float:
            nonlocal proactive_cancel_tree_build_overlap_ms_sum
            nonlocal proactive_cancel_tree_build_overlap_ms_count
            nonlocal proactive_cancel_tree_build_overlap_ms_max
            nonlocal proactive_cancel_tree_build_overlap_event_count
            if not _pending_canceled_proactive:
                return 0.0
            try:
                build_start = float(build_start_ts)
                build_end = float(build_end_ts)
            except Exception:
                return 0.0
            if build_end <= build_start:
                return 0.0
            total_overlap_ms = 0.0
            overlap_events = 0
            for rec in _pending_canceled_proactive:
                state = rec.get("state") or {}
                th = rec.get("thread")
                cancel_ts = rec.get("cancel_ts")
                if cancel_ts is None:
                    continue
                exit_ts = state.get("thread_exited_ts")
                if exit_ts is None:
                    if th is None or not th.is_alive():
                        continue
                    exit_ts = build_end
                try:
                    active_start = max(float(cancel_ts), build_start)
                    active_end = min(float(exit_ts), build_end)
                except Exception:
                    continue
                overlap_ms = max(0.0, (active_end - active_start) * 1000.0)
                if overlap_ms > 0:
                    total_overlap_ms += overlap_ms
                    overlap_events += 1
            if total_overlap_ms > 0:
                proactive_cancel_tree_build_overlap_ms_sum += total_overlap_ms
                proactive_cancel_tree_build_overlap_ms_count += 1
                proactive_cancel_tree_build_overlap_ms_max = max(
                    proactive_cancel_tree_build_overlap_ms_max,
                    total_overlap_ms,
                )
                proactive_cancel_tree_build_overlap_event_count += overlap_events
            return total_overlap_ms

        def _join_canceled_proactive_before_tree_build() -> float:
            nonlocal proactive_cancel_join_before_tree_build_ms_sum
            nonlocal proactive_cancel_join_before_tree_build_ms_count
            nonlocal proactive_cancel_join_before_tree_build_ms_max
            if not join_canceled_proactive_before_tree_build or not _pending_canceled_proactive:
                return 0.0
            join_start = time.time()
            _harvest_canceled_proactive_threads(force_join=True)
            join_ms = max(0.0, (time.time() - join_start) * 1000.0)
            if join_ms > 0:
                proactive_cancel_join_before_tree_build_ms_sum += join_ms
                proactive_cancel_join_before_tree_build_ms_count += 1
                proactive_cancel_join_before_tree_build_ms_max = max(
                    proactive_cancel_join_before_tree_build_ms_max,
                    join_ms,
                )
            return join_ms

        def _join_canceled_proactive_immediately() -> float:
            nonlocal proactive_cancel_immediate_join_ms_sum
            nonlocal proactive_cancel_immediate_join_ms_count
            nonlocal proactive_cancel_immediate_join_ms_max
            if not join_canceled_proactive_before_tree_build or not _pending_canceled_proactive:
                return 0.0
            join_start = time.time()
            _harvest_canceled_proactive_threads(force_join=True)
            join_ms = max(0.0, (time.time() - join_start) * 1000.0)
            if join_ms > 0:
                proactive_cancel_immediate_join_ms_sum += join_ms
                proactive_cancel_immediate_join_ms_count += 1
                proactive_cancel_immediate_join_ms_max = max(
                    proactive_cancel_immediate_join_ms_max,
                    join_ms,
                )
            return join_ms

        def _account_proactive_compute(
            state: dict,
            *,
            used: bool,
            post_reply_wait_ms: float = 0.0,
        ) -> float:
            nonlocal proactive_total_time_sec
            nonlocal proactive_wall_elapsed_total_sec
            nonlocal proactive_compute_used_sec
            nonlocal proactive_compute_unused_sec
            nonlocal proactive_compute_hidden_used_sec
            nonlocal proactive_compute_after_reply_used_sec
            if not state or state.get("compute_accounted", False):
                return float(state.get("compute_elapsed_sec", 0.0) or 0.0) if state else 0.0
            elapsed_sec = state.get("elapsed_sec")
            if elapsed_sec is None:
                return 0.0
            budget_wait_sec = max(
                0.0,
                float(state.get("budget_wait_ms", 0.0) or 0.0) / 1000.0,
            )
            wall_elapsed_sec = max(0.0, float(elapsed_sec))
            compute_elapsed_sec = max(0.0, wall_elapsed_sec - budget_wait_sec)
            post_reply_compute_sec = min(
                compute_elapsed_sec,
                max(0.0, float(post_reply_wait_ms or 0.0) / 1000.0),
            )
            hidden_used_sec = (
                max(0.0, compute_elapsed_sec - post_reply_compute_sec)
                if used
                else 0.0
            )

            state["compute_elapsed_sec"] = compute_elapsed_sec
            state["compute_after_reply_sec"] = post_reply_compute_sec if used else 0.0
            state["compute_hidden_before_reply_sec"] = hidden_used_sec
            state["compute_accounted"] = True
            if isinstance(state.get("gpu_stats"), dict) and compute_elapsed_sec > 0:
                proactive_gpu_data.append({
                    "step": state.get("source_step"),
                    "timestamp": datetime.now().isoformat(),
                    "used": bool(used),
                    "total_time_seconds": float(compute_elapsed_sec),
                    "gpu_stats": state.get("gpu_stats"),
                    "gpu_power_avg_w": float(state.get("gpu_power_avg_w", 0.0) or 0.0),
                    "gpu_energy_joules": float(state.get("gpu_energy_joules", 0.0) or 0.0),
                    "gpu_energy_kwh": float(state.get("gpu_energy_kwh", 0.0) or 0.0),
                })
                proactive_timing_data.append({
                    "total_time_seconds": float(compute_elapsed_sec),
                    "timestamp": datetime.now().isoformat(),
                    "used": bool(used),
                })

            proactive_wall_elapsed_total_sec += wall_elapsed_sec
            proactive_total_time_sec += compute_elapsed_sec
            if used:
                proactive_compute_used_sec += compute_elapsed_sec
                proactive_compute_after_reply_used_sec += post_reply_compute_sec
                proactive_compute_hidden_used_sec += hidden_used_sec
            else:
                proactive_compute_unused_sec += compute_elapsed_sec
            return compute_elapsed_sec

        def _harvest_canceled_proactive_threads(force_join: bool = False):
            nonlocal proactive_cancel_to_exit_ms_sum
            nonlocal proactive_cancel_to_exit_ms_count
            nonlocal proactive_cancel_to_exit_ms_max
            nonlocal proactive_cancel_to_exit_recent_ms
            nonlocal proactive_cancel_pending_count
            if not _pending_canceled_proactive:
                proactive_cancel_pending_count = 0
                return
            remaining = []
            for rec in _pending_canceled_proactive:
                th = rec.get("thread")
                state = rec.get("state") or {}
                cancel_ts = rec.get("cancel_ts")
                if th is None or cancel_ts is None:
                    continue
                if force_join:
                    th.join()
                if th.is_alive():
                    remaining.append(rec)
                    continue
                exit_ts = state.get("thread_exited_ts")
                if exit_ts is None:
                    exit_ts = time.time()
                try:
                    cancel_to_exit_ms = max(0.0, (float(exit_ts) - float(cancel_ts)) * 1000.0)
                except Exception:
                    cancel_to_exit_ms = 0.0
                proactive_cancel_to_exit_ms_sum += float(cancel_to_exit_ms)
                proactive_cancel_to_exit_ms_count += 1
                proactive_cancel_to_exit_ms_max = max(
                    proactive_cancel_to_exit_ms_max, float(cancel_to_exit_ms)
                )
                proactive_cancel_to_exit_recent_ms.append(float(cancel_to_exit_ms))
                if len(proactive_cancel_to_exit_recent_ms) > 256:
                    del proactive_cancel_to_exit_recent_ms[:-256]
                _account_proactive_compute(state, used=False, post_reply_wait_ms=0.0)
            _pending_canceled_proactive[:] = remaining
            proactive_cancel_pending_count = len(_pending_canceled_proactive)

        def _estimate_proactive_cancel_loss_sec() -> float:
            """Estimate direct cancellation tail risk; post-reply wait is intentionally excluded."""
            if proactive_cancel_to_exit_recent_ms:
                arr = sorted(float(v) for v in proactive_cancel_to_exit_recent_ms if v is not None)
                if arr:
                    idx = int(min(len(arr) - 1, max(0, round(0.95 * (len(arr) - 1)))))
                    return max(0.0, arr[idx] / 1000.0)
            if proactive_cancel_to_exit_ms_count > 0:
                return max(
                    0.0,
                    float(proactive_cancel_to_exit_ms_sum)
                    / float(proactive_cancel_to_exit_ms_count)
                    / 1000.0,
                )
            # Warmup floor keeps cs0 from starting near-zero-probability paths before samples exist.
            return 0.005

        def _refresh_proactive_budget():
            nonlocal proactive_budget_ms_latest
            if disable_proactive_budget:
                proactive_budget_ms_latest = None
                runner.proactive_budget_ms = None
                return
            if (
                last_target_verification_ms is None
                or last_draft_to_target_ms is None
                or last_target_to_draft_ms is None
            ):
                proactive_budget_ms_latest = None
                runner.proactive_budget_ms = None
                return
            proactive_budget_ms_latest = max(
                0.0,
                float(last_target_verification_ms)
                + float(last_draft_to_target_ms)
                + float(last_target_to_draft_ms),
            )
            runner.proactive_budget_ms = float(proactive_budget_ms_latest)

        def _record_proactive_depth_outcome(depth_value, outcome: str):
            try:
                depth_i = int(depth_value)
            except Exception:
                return
            rec = proactive_depth_stats.setdefault(depth_i, {"used": 0, "canceled": 0})
            key = "used" if outcome == "used" else "canceled"
            rec[key] = int(rec.get(key, 0) or 0) + 1

        def _merge_proactive_expand_metrics(state: dict):
            nonlocal proactive_finalize_early_count
            nonlocal proactive_expand_continue_count
            nonlocal proactive_expand_pause_count
            nonlocal proactive_expected_gain_ms_sum
            nonlocal proactive_expected_loss_ms_sum
            if not isinstance(state, dict):
                return
            proactive_finalize_early_count += int(state.get("finalize_early_count", 0) or 0)
            proactive_expand_continue_count += int(state.get("expand_continue_count", 0) or 0)
            proactive_expand_pause_count += int(state.get("expand_pause_count", 0) or 0)
            proactive_expected_gain_ms_sum += float(state.get("expected_gain_ms", 0.0) or 0.0)
            proactive_expected_loss_ms_sum += float(state.get("expected_loss_ms", 0.0) or 0.0)
            for depth_key, count in (state.get("expand_depth_counts", {}) or {}).items():
                try:
                    depth_i = int(depth_key)
                    count_i = int(count)
                except Exception:
                    continue
                proactive_expand_depth_counts[depth_i] = proactive_expand_depth_counts.get(depth_i, 0) + count_i

        def _update_post_reply_bucket(bucket: dict, key, post_wait_ms: float):
            rec = bucket.setdefault(
                key,
                {"count": 0, "total_ms": 0.0, "max_ms": 0.0, "positive_count": 0},
            )
            val = max(0.0, float(post_wait_ms or 0.0))
            rec["count"] += 1
            rec["total_ms"] += val
            rec["max_ms"] = max(float(rec.get("max_ms", 0.0)), val)
            if val > 0:
                rec["positive_count"] += 1

        def _record_proactive_post_reply_shape(tree_obj: dict, post_wait_ms: float):
            if not isinstance(tree_obj, dict):
                return None
            depth = tree_obj.get("tree_depth")
            widths = tree_obj.get("depth_widths") or []
            final_nodes = tree_obj.get("final_nnodes")
            try:
                depth_i = int(depth)
            except Exception:
                depth_i = None
            try:
                avg_width = float(sum(float(w) for w in widths) / len(widths)) if widths else None
            except Exception:
                avg_width = None
            try:
                final_nodes_i = int(final_nodes)
            except Exception:
                final_nodes_i = None
            width_bin = int(round(avg_width / 10.0) * 10) if avg_width is not None else None
            if depth_i is not None:
                _update_post_reply_bucket(proactive_post_reply_by_depth, str(depth_i), post_wait_ms)
            if width_bin is not None:
                _update_post_reply_bucket(proactive_post_reply_by_width, str(width_bin), post_wait_ms)
            if depth_i is not None and width_bin is not None:
                _update_post_reply_bucket(
                    proactive_post_reply_by_depth_width,
                    f"d{depth_i}_w{width_bin}",
                    post_wait_ms,
                )
            return {
                "depth": depth_i,
                "avg_width": avg_width,
                "width_bin": width_bin,
                "final_nnodes": final_nodes_i,
            }

        def _target_energy_kwh_from_step(target_verification_time_ms, target_energy_rate_per_sec):
            target_sec = max(0.0, float(target_verification_time_ms or 0.0) / 1000.0)
            if target_sec <= 0:
                return 0.0, False
            try:
                target_energy_rate = float(target_energy_rate_per_sec)
            except Exception:
                target_energy_rate = 0.0
            if target_energy_rate <= 0:
                return 0.0, False
            return max(0.0, target_energy_rate * target_sec), True

        def _estimate_step_metric_spend(
            tree_build_time_ms,
            target_verification_time_ms,
            d2t_step_bytes,
            t2d_step_bytes,
            gpu_stats_step=None,
            target_energy_rate_per_sec=None,
            server_only_step=False,
            server_only_wall_time_ms=None,
            server_draft_energy_rate_per_sec=None,
        ) -> float:
            def _extract_total_gpu_power_w(stats: dict) -> float:
                if not isinstance(stats, dict):
                    return 0.0
                total_w = 0.0
                for gpu_entry in stats.values():
                    if not isinstance(gpu_entry, dict):
                        continue
                    power_info = gpu_entry.get("power_draw_w", {})
                    power_avg_w = power_info.get("avg") if isinstance(power_info, dict) else None
                    if power_avg_w is None:
                        continue
                    try:
                        power_avg_w = float(power_avg_w)
                    except Exception:
                        continue
                    if power_avg_w > 0:
                        total_w += power_avg_w
                return total_w

            if objective_metric in cost_objective_metrics:
                draft_sec = max(0.0, float(tree_build_time_ms or 0.0) / 1000.0)
                target_sec = max(0.0, float(target_verification_time_ms or 0.0) / 1000.0)
                target_cost = target_sec * float(target_per_sec_cost)
                inbound_cost = (
                    (max(0.0, float(d2t_step_bytes or 0.0)) / float(1024 ** 3))
                    * float(user_communication_cost_per_gb)
                )
                outbound_cost = (
                    (max(0.0, float(t2d_step_bytes or 0.0)) / float(1024 ** 3))
                    * float(user_communication_cost_per_gb + cloud_outbound_cost_per_gb)
                )
                if objective_metric == "api_cost":
                    if server_only_step:
                        return max(0.0, (draft_sec + target_sec) * float(target_per_sec_cost))
                    return max(0.0, target_cost + outbound_cost)
                if no_draft_cost:
                    draft_cost_step = 0.0
                else:
                    if objective_metric == "total_cost":
                        if bool(getattr(runner, "bill_draft_as_target_gpu", False)):
                            draft_cost_step = draft_sec * float(target_per_sec_cost)
                        else:
                            draft_power_w = _extract_total_gpu_power_w(gpu_stats_step)
                            if draft_power_w > 0:
                                draft_cost_step = (
                                    (draft_power_w * draft_sec) / 3600000.0
                                ) * float(runner.draft_electricity_cost_per_kwh)
                            else:
                                draft_cost_step = draft_sec * float(runner.get_draft_objective_rate_per_sec())
                    else:
                        draft_cost_step = draft_sec * float(draft_per_sec_cost)
                if objective_metric == "total_cost" and server_only_step:
                    # server-only total_cost step wall-time .
                    if server_only_wall_time_ms is not None:
                        server_wall_sec = max(0.0, float(server_only_wall_time_ms) / 1000.0)
                        return max(0.0, server_wall_sec * float(target_per_sec_cost))
                    # fallback: wall-time
                    return max(0.0, draft_cost_step + target_cost)
                return max(0.0, draft_cost_step + target_cost + inbound_cost + outbound_cost)

            total_metric = 0.0
            include_draft_energy = objective_metric == "draft_energy" or (
                server_only_step and objective_metric == "target_energy"
            )
            if include_draft_energy:
                draft_sec = max(0.0, float(tree_build_time_ms or 0.0) / 1000.0)
                if draft_sec > 0:
                    draft_power_w = _extract_total_gpu_power_w(gpu_stats_step)
                    if draft_power_w > 0:
                        total_metric += (draft_power_w * draft_sec) / 3600000.0
                    elif server_only_step and server_draft_energy_rate_per_sec is not None:
                        try:
                            server_draft_energy_rate = float(server_draft_energy_rate_per_sec)
                        except Exception:
                            server_draft_energy_rate = 0.0
                        total_metric += max(0.0, server_draft_energy_rate) * draft_sec
                    else:
                        total_metric += float(runner.get_draft_objective_rate_per_sec()) * draft_sec

            if objective_metric == "target_energy":
                target_sec = max(0.0, float(target_verification_time_ms or 0.0) / 1000.0)
                if target_sec > 0:
                    try:
                        target_energy_rate_per_sec = float(target_energy_rate_per_sec)
                    except Exception:
                        target_energy_rate_per_sec = None
                    if target_energy_rate_per_sec is not None and target_energy_rate_per_sec > 0:
                        total_metric += target_energy_rate_per_sec * target_sec
                    else:
                        total_metric += float(runner.get_target_objective_rate_per_sec()) * target_sec

            return max(0.0, total_metric)

        # Network latency and accept-length records
        network_latency_records = []
        # Query/reference update history for UI reporting
        reference_update_history = []
        global_draft_to_target_times = []
        global_target_to_draft_times = []
        global_accept_lengths = []
        global_t2d_per_accept_len = {}
        global_d2t_per_final_nnodes = {}  # final_nnodes draft_to_target_time

        choices_list = []
        all_answers = []
        tree_width_records = []  # depth width
        
        # Latency
        latency_data = []  # step latency
        # run 
        run_draft_to_target_bytes = []
        run_target_to_draft_bytes = []
        # online profile update 
        profile_update_overhead_sec_total = 0.0
        # objective target-side energy . target server
        # GPU monitor target_energy_rate_per_sec .
        total_target_energy_kwh = 0.0
        target_energy_sample_count = 0
        target_energy_missing_count = 0
        
        # Accept length (accept_stats )
        all_accept_lengths = []  # step accept_length
        # Expected accept length (sum_expected_accepted_length) : accept_length bucket
        all_expected_accept_lengths_per_step = []  # step expected accept length (tree.sum_expected_accepted_length)
        all_scaled_expected_accept_lengths_per_step = []  # step scaled expected accept length (expected * accept_length_scale_used)
        all_clipped_expected_accept_lengths_per_step = []  # step clipped expected accept length (scaled * (1-accept_length_margin))
        # step predicted-vs-actual
        accept_length_step_pairs = {
            "actual_accept_length_per_step": [],
            "expected_accept_length_raw_per_step": [],
            "expected_accept_length_scaled_per_step": [],
            "expected_accept_length_clipped_per_step": [],
        }
        
        # Number of nodes (avg_number_of_nodes )
        all_final_nnodes = []  # step final_nnodes
        
        # GPU
        draft_gpu_data = []  # step GPU
        draft_timing_data = []  # step timing 
        proactive_gpu_data = []  # proactive draft tree compute GPU
        proactive_timing_data = []  # proactive GPU
        # CPU
        draft_cpu_power_data = []  # step CPU
        
        for question in tqdm(questions):
            if metric_cap_reached:
                break
            # breakpoint()
            question_start_time = time.time()
            choices = []
            draft_to_target_time_per_question = []
            target_to_draft_time_per_question = []
            accept_length_per_question = []
            t2d_per_accept_len_per_question = {}
            d2t_per_final_nnodes_per_question = {}  # final_nnodes draft_to_target_time
            target_verify_per_nnodes_per_question = {}  # nnodes target verification (ms)
            # Per-question transfer byte counters
            draft_to_target_bytes_per_question = []
            target_to_draft_bytes_per_question = []
            draft_model_call_ms_per_width_per_question = {}

            
            for i in range(num_choices):
                if metric_cap_reached:
                    break
                torch.manual_seed(i)
                # question KV cache
                runner.reset_kv()
                
                conv = _build_conversation_template_for_model(base_model_path)

                question_turns = _extract_question_turns(question, bench_name)

                turns = []
                idxs = []
                new_tokens_list = []
                wall_time_list = []
                for j in range(len(question_turns)):
                    if metric_cap_reached:
                        break
                    qs = question_turns[j]
                    conv.append_message(conv.roles[0], qs)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt() + " "
                    input_ids = tokenizer([prompt]).input_ids
                    input_ids_t = torch.as_tensor(input_ids).to("cuda")


                    if server_only_mode:
                        draft_send_start_time = time.time()
                        prev_sock_timeout = sock.gettimeout()
                        if bool(force_server_only_ar):
                            print(
                                f"[ServerOnlyAR] init send begin prompt_tokens={len(input_ids[0])} "
                                f"max_new_tokens={int(server_only_ar_max_new_tokens)}",
                                flush=True,
                            )
                        init_timeout_s = (
                            30.0
                            if bool(force_server_only_ar)
                            else _estimate_server_only_init_timeout_seconds(
                                server_draft_profile_auto=bool(server_draft_profile_auto),
                                server_draft_profile_force_refresh=bool(server_draft_profile_force_refresh),
                                server_draft_profile_model_calls_per_count=int(max(1, int(server_draft_profile_model_calls_per_count))),
                                server_draft_profile_width_list=str(server_draft_profile_width_list),
                            )
                        )
                        sock.settimeout(init_timeout_s)
                        server_only_init_payload = {
                            "type": (
                                "server_only_ar_turn"
                                if bool(force_server_only_ar) and bool(server_only_ar_turn_rpc)
                                else ("server_only_ar_init" if bool(force_server_only_ar) else "server_only_init")
                            ),
                            "input_ids": input_ids[0],
                            "max_new_tokens": int(server_only_ar_max_new_tokens),
                            "nodes": int(nodes),
                            "max_depth": int(max_depth),
                            "per_token_probability_bound": float(per_token_probability_bound),
                            "per_path_probability_bound": float(per_path_probability_bound),
                            "min_width": int(min_width),
                            "fixed_width": bool(fixed_width),
                            "fixed_width_value": fixed_width_value,
                            "fixed_nnodes": bool(fixed_nnodes),
                            "fixed_depth": bool(fixed_depth),
                            "proactive_drafting": bool(proactive_drafting),
                            "proactive_threshold": float(proactive_threshold),
                            "adaptive_proactive_threshold": bool(adaptive_proactive_threshold),
                            "cost_sensitivity": float(cost_sensitivity),
                            "draft_per_sec_cost": float(draft_per_sec_cost),
                            "target_per_sec_cost": float(target_per_sec_cost),
                            "bill_draft_as_target_gpu": bool(getattr(runner, "bill_draft_as_target_gpu", False)),
                            "server_draft_profile_auto": bool(server_draft_profile_auto),
                            "server_draft_profile_force_refresh": bool(server_draft_profile_force_refresh)
                            and not bool(server_draft_profile_force_refresh_consumed),
                            "server_draft_profile_model_calls_per_count": int(max(1, int(server_draft_profile_model_calls_per_count))),
                            "server_draft_profile_width_list": str(server_draft_profile_width_list),
                            "draft_model_path": str(draft_model_path),
                            "draft_quantization": str(draft_quantization),
                            "server_name": str(server_name),
                            "bench_name": str(bench_name),
                            "reference_tps": float(runner.reference_tps),
                            "reference_objective_per_token": float(runner.reference_objective_per_token),
                            "objective_metric": str(objective_metric),
                            "objective_selection_mode": str(objective_selection_mode),
                            "metric_constraint_per_token": (
                                float(metric_constraint_per_token)
                                if metric_constraint_per_token is not None
                                else None
                            ),
                            "no_draft_cost": bool(no_draft_cost),
                        }
                        d2t_bytes = send_json_with_size(sock, server_only_init_payload)
                        if bool(force_server_only_ar):
                            print(
                                "[ServerOnlyAR] init sent; waiting for server_only_ok "
                                f"mode={server_only_init_payload['type']}",
                                flush=True,
                            )
                        try:
                            reply, t2d_bytes = recv_json_with_size(sock)
                            if bool(force_server_only_ar):
                                print(f"[ServerOnlyAR] init reply type={reply.get('type')}", flush=True)
                            if (
                                reply.get("type") == "error"
                                and "target model is not loaded" in str(reply.get("message", ""))
                            ):
                                print(
                                    "[Startup] target reported unloaded during server-only init; reloading and retrying once",
                                    flush=True,
                                )
                                _ensure_remote_target_model(
                                    sock=sock,
                                    base_model_path=base_model_path,
                                    load_in_4bit=load_in_4bit,
                                    load_in_8bit=load_in_8bit,
                                    target_quantization=target_quantization,
                                    device_map=device_map,
                                    debug=debug,
                                )
                                d2t_retry_bytes = send_json_with_size(sock, server_only_init_payload)
                                reply, t2d_bytes = recv_json_with_size(sock)
                                d2t_bytes += d2t_retry_bytes
                                if bool(force_server_only_ar):
                                    print(f"[ServerOnlyAR] retry init reply type={reply.get('type')}", flush=True)
                        finally:
                            sock.settimeout(prev_sock_timeout)
                        draft_recv_end_time = time.time()
                    else:
                        # base 1 target
                        draft_send_start_time = time.time()
                        d2t_bytes = send_json_with_size(sock, {"type": "init", "input_ids": input_ids[0]})
                        reply, t2d_bytes = recv_json_with_size(sock)
                        draft_recv_end_time = time.time()
                    
                    # Per-question transfer byte counters
                    draft_to_target_bytes_per_question.append(d2t_bytes)
                    target_to_draft_bytes_per_question.append(t2d_bytes)
                    run_draft_to_target_bytes.append(d2t_bytes)
                    run_target_to_draft_bytes.append(t2d_bytes)

                    target_recv_end_time = reply.get("target_recv_end_time", None)
                    target_send_start_time = reply.get("target_send_start_time", None)
                    # Draft Target
                    draft_to_target_time = None
                    if target_recv_end_time is not None :
                        draft_to_target_time = max(0.05, (target_recv_end_time - draft_send_start_time) * 1000.0)  # ms
                        draft_to_target_time_per_question.append(draft_to_target_time)
    
                    # Target Draft
                    target_to_draft_time = None
                    if target_send_start_time is not None:
                        target_to_draft_time = max(0.05, (draft_recv_end_time - target_send_start_time) * 1000.0)  # ms
                        target_to_draft_time_per_question.append(target_to_draft_time)
                        if 1 not in t2d_per_accept_len_per_question:
                            t2d_per_accept_len_per_question[0] = []
                        t2d_per_accept_len_per_question[0].append(target_to_draft_time)
                    
                    accept_length_per_question.append(0)

                    
                    if server_only_mode:
                        reply_type = reply.get("type")
                        if reply_type != "server_only_ok":
                            raise RuntimeError(
                                "server_only_init failed: "
                                f"type={reply_type}, message={reply.get('message')}, reply={reply}"
                            )
                        profile_status = reply.get("server_draft_profile", {})
                        if isinstance(profile_status, dict):
                            server_only_profile_prepare_overhead_sec += float(
                                max(0.0, float(profile_status.get("prepare_wall_sec", 0.0) or 0.0))
                            )
                        if bool(server_only_init_payload.get("server_draft_profile_force_refresh", False)):
                            server_draft_profile_force_refresh_consumed = True
                        current_next_token = None
                        if bool(force_server_only_ar):
                            # AR prefill on the target can take longer than the
                            # regular RPC timeout before the first token/turn reply.
                            # Keep a finite timeout so a target-side CUDA/RPC stall
                            # fails this run instead of blocking the whole sweep.
                            ar_stream_timeout_s = float(
                                os.environ.get("SERVER_ONLY_AR_STREAM_TIMEOUT_SEC", "300.0")
                            )
                            sock.settimeout(ar_stream_timeout_s if ar_stream_timeout_s > 0 else None)
                            print(
                                f"[ServerOnlyAR] receive begin mode={server_only_init_payload['type']} "
                                f"timeout={ar_stream_timeout_s}s",
                                flush=True,
                            )
                    else:
                        assert reply.get("type") == "init_ok"
                        current_next_token = reply["next_token"]

                    output_tokens: List[int] = []
                    new_token_count = 0
                    turn_steps = 0
                    pending_proactive = None
                    use_proactive_tree = False
                    proactive_tree = None
                    prev_step_used_proactive_tree = False
                    prev_step_proactive_canceled = False
                    
                    if not bool(force_server_only_ar):
                        torch.cuda.synchronize()
                    turn_start_time = time.time()

                    while True:
                        if metric_cap_reached:
                            break
                        if server_only_mode:
                            try:
                                reply, t2d_bytes = recv_json_with_size(sock)
                            except TimeoutError:
                                if bool(force_server_only_ar):
                                    print(
                                        "[ServerOnlyAR][TIMEOUT] recv timed out "
                                        f"question_id={question.get('question_id')} "
                                        f"choice_idx={i} "
                                        f"received_tokens={new_token_count} "
                                        f"turn_steps={turn_steps} "
                                        f"prompt_tokens={len(input_ids[0])} "
                                        f"max_new_tokens={int(server_only_ar_max_new_tokens)} "
                                        f"socket_timeout={sock.gettimeout()}",
                                        flush=True,
                                    )
                                raise
                            draft_recv_end_time = time.time()
                            # server-only step Target -> Draft
                            target_to_draft_bytes_per_question.append(t2d_bytes)
                            run_target_to_draft_bytes.append(t2d_bytes)
                            if reply.get("type") == "server_only_done":
                                break
                            if reply.get("type") != "verify_result":
                                raise RuntimeError(
                                    "server-only step failed: "
                                    f"type={reply.get('type')}, message={reply.get('message')}, reply={reply}"
                                )
                            accepted_tokens: List[int] = reply["accepted_tokens"]
                            accept_length: int = reply["accept_length"]
                            next_token: int = reply["next_token"]
                            eos_reached: bool = reply["eos_reached"]
                            ar_emitted_tokens = new_token_count + len(accepted_tokens)
                            ar_done_inline = bool(reply.get("server_only_done", False))
                            ar_client_log_interval = int(
                                max(1, int(os.environ.get("SERVER_ONLY_AR_CLIENT_LOG_INTERVAL", "32")))
                            )
                            if bool(force_server_only_ar) and (
                                new_token_count == 0
                                or ar_emitted_tokens % ar_client_log_interval == 0
                                or bool(eos_reached)
                                or bool(ar_done_inline)
                            ):
                                print(
                                    "[ServerOnlyAR] received "
                                    f"tokens={ar_emitted_tokens} "
                                    f"eos={bool(eos_reached)} "
                                    f"done={bool(ar_done_inline)} "
                                    f"limit={bool(reply.get('generation_limit_reached', False))}",
                                    flush=True,
                                )
                            target_verification_time = reply.get("target_verification_time_ms", None)
                            target_energy_rate_per_sec = reply.get("target_energy_rate_per_sec", None)
                            tree_build_time = reply.get("tree_build_time_ms", 0.0)
                            draft_to_target_time = reply.get("draft_to_target_time_ms", None)
                            target_to_draft_time = reply.get("target_to_draft_time_ms", None)
                            tree_model_forward_ms = float(reply.get("tree_model_forward_ms", 0.0) or 0.0)
                            tree_width_algo_ms = float(reply.get("tree_width_algo_ms", 0.0) or 0.0)
                            tree_nnodes_algo_ms = float(reply.get("tree_nnodes_algo_ms", 0.0) or 0.0)
                            tree_mask_build_ms = float(reply.get("tree_mask_build_ms", 0.0) or 0.0)
                            tree_finalize_ms = float(reply.get("tree_finalize_ms", 0.0) or 0.0)
                            if draft_to_target_time is not None:
                                draft_to_target_time = max(0.05, float(draft_to_target_time))
                            if target_to_draft_time is not None:
                                target_to_draft_time = max(0.05, float(target_to_draft_time))
                            final_nnodes = int(reply.get("final_nnodes", 0))
                            tree_depth = int(reply.get("tree_depth", 0))
                            depth_widths = reply.get("depth_widths", [])
                            if draft_to_target_time is not None:
                                draft_to_target_time_per_question.append(draft_to_target_time)
                            if target_to_draft_time is not None:
                                target_to_draft_time_per_question.append(target_to_draft_time)
                            reported_accept_length = 1 if bool(force_server_only_ar) else int(accept_length)
                            accept_length_per_question.append(reported_accept_length)
                            all_final_nnodes.append(final_nnodes)
                            if isinstance(depth_widths, list) and depth_widths:
                                avg_width = float(sum(depth_widths) / len(depth_widths))
                                tree_width_records.append({
                                    "question_id": question["question_id"],
                                    "choice_idx": i,
                                    "step": turn_steps,
                                    "depth_widths": [int(w) for w in depth_widths],
                                    "avg_width": avg_width,
                                    "final_nnodes": int(final_nnodes),
                                    "tree_depth": int(tree_depth),
                                })
                            if target_verification_time is not None:
                                last_target_verification_ms = float(target_verification_time)
                                tvals = target_verify_per_nnodes_per_question.setdefault(int(final_nnodes), [])
                                tvals.append(float(target_verification_time))
                            step_target_energy_kwh, step_target_energy_measured = _target_energy_kwh_from_step(
                                target_verification_time,
                                target_energy_rate_per_sec,
                            )
                            total_target_energy_kwh += float(step_target_energy_kwh)
                            if step_target_energy_measured:
                                target_energy_sample_count += 1
                            else:
                                target_energy_missing_count += 1
                            runner.update_target_objective_rate(target_energy_rate_per_sec)
                            if draft_to_target_time is not None:
                                last_draft_to_target_ms = float(draft_to_target_time)
                            if target_to_draft_time is not None:
                                last_target_to_draft_ms = float(target_to_draft_time)
                            last_tree_build_ms = float(tree_build_time)
                            _refresh_proactive_budget()
                            exp_raw = reply.get("expected_accept_length", None)
                            exp_scaled = reply.get("expected_accept_length_scaled", None)
                            exp_clipped = reply.get("expected_accept_length_clipped", None)
                            step_latency = {
                                "tree_build_time_ms": tree_build_time,
                                "draft_to_target_time_ms": draft_to_target_time,
                                "target_verification_time_ms": target_verification_time,
                                "target_to_draft_time_ms": target_to_draft_time,
                                "tree_model_forward_ms": tree_model_forward_ms,
                                "tree_width_algo_ms": tree_width_algo_ms,
                                "tree_nnodes_algo_ms": tree_nnodes_algo_ms,
                                "tree_mask_build_ms": tree_mask_build_ms,
                                "tree_finalize_ms": tree_finalize_ms,
                                "tree_budget_wait_ms": 0.0,
                                # server-only step draft->target payload 0
                                "draft_to_target_bytes": 0,
                                "target_to_draft_bytes": t2d_bytes,
                                "expected_accept_length": exp_raw,
                                "proactive_budget_ms": proactive_budget_ms_latest,
                                "proactive_budget_wait_ms": 0.0,
                                "proactive_resume_after_reply_ms": 0.0,
                                "proactive_post_reply_wait_ms": 0.0,
                                "proactive_head_ready_at_reply": False,
                                "proactive_final_ready_at_reply": False,
                                "proactive_cancel_reason": None,
                                "proactive_cancel_to_exit_ms": None,
                                "total_time_ms": (tree_build_time or 0.0)
                                + (draft_to_target_time or 0.0)
                                + (target_verification_time or 0.0)
                                + (target_to_draft_time or 0.0),
                            }
                            latency_data.append(step_latency)
                            output_tokens.extend(accepted_tokens)
                            new_token_count += len(accepted_tokens)
                            turn_steps += 1
                            # server-only verify_result step/token .
                            grand_total_steps += 1
                            grand_total_accepted += int(accept_length)
                            grand_total_draft_tokens += int(tree_depth)
                            all_accept_lengths.append(reported_accept_length)
                            all_expected_accept_lengths_per_step.append(exp_raw)
                            all_scaled_expected_accept_lengths_per_step.append(exp_scaled)
                            all_clipped_expected_accept_lengths_per_step.append(exp_clipped)
                            accept_length_step_pairs["actual_accept_length_per_step"].append(reported_accept_length)
                            accept_length_step_pairs["expected_accept_length_raw_per_step"].append(exp_raw)
                            accept_length_step_pairs["expected_accept_length_scaled_per_step"].append(exp_scaled)
                            accept_length_step_pairs["expected_accept_length_clipped_per_step"].append(exp_clipped)
                            step_total_time_ms = (
                                float(tree_build_time or 0.0)
                                + float(draft_to_target_time or 0.0)
                                + float(target_verification_time or 0.0)
                                + float(target_to_draft_time or 0.0)
                            )
                            step_metric = _estimate_step_metric_spend(
                                tree_build_time_ms=tree_build_time,
                                target_verification_time_ms=target_verification_time,
                                d2t_step_bytes=0.0,
                                t2d_step_bytes=t2d_bytes,
                                gpu_stats_step=None,
                                target_energy_rate_per_sec=target_energy_rate_per_sec,
                                server_only_step=True,
                                server_only_wall_time_ms=step_total_time_ms,
                                server_draft_energy_rate_per_sec=reply.get("server_draft_energy_rate_per_sec", None),
                            )
                            step_tokens_for_ref = float(int(accept_length) + 1)
                            if step_total_time_ms > 0 and step_tokens_for_ref > 0:
                                ref_update_total_time_sec += float(step_total_time_ms / 1000.0)
                                ref_update_total_tokens += float(step_tokens_for_ref)
                                ref_update_total_metric += float(max(0.0, float(step_metric)))
                            metric_spent_total += float(step_metric)
                            if np.isfinite(total_metric_cap) and metric_spent_total >= float(total_metric_cap):
                                metric_cap_reached = True
                                if debug:
                                    print(
                                        f"[DRAFT-DEBUG] metric cap reached in server-only mode: "
                                        f"spent={metric_spent_total:.6f}, cap={float(total_metric_cap):.6f}"
                                    )
                                break
                            if eos_reached or new_token_count >= int(max_new_tokens):
                                if bool(force_server_only_ar):
                                    if bool(reply.get("server_only_done", False)):
                                        break
                                    prev_done_timeout = sock.gettimeout()
                                    try:
                                        sock.settimeout(float(os.environ.get("SERVER_ONLY_AR_DONE_TIMEOUT_SEC", "10.0")))
                                        done_reply, done_bytes = recv_json_with_size(sock)
                                        target_to_draft_bytes_per_question.append(done_bytes)
                                        run_target_to_draft_bytes.append(done_bytes)
                                        if done_reply.get("type") != "server_only_done":
                                            raise RuntimeError(
                                                "server-only AR expected server_only_done after final token, "
                                                f"got type={done_reply.get('type')}, reply={done_reply}"
                                            )
                                    finally:
                                        sock.settimeout(prev_done_timeout)
                                    break
                                # server-only verify_result
                                # server_only_done ,
                                # break done .
                                continue
                            continue
                        proactive_cancel_join_before_tree_build_ms = _join_canceled_proactive_before_tree_build()

                        # GPU (build_tree_with_next_token )
                        gpu_monitor_interval_start = time.time()
                        if runner.gpu_monitor and not gpu_monitor_long_running:
                            runner.gpu_monitor.start_monitoring()
                        # CPU (GPU )
                        if runner.cpu_power_monitor:
                            runner.cpu_power_monitor.start_monitoring()
                        
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        tree_build_start_time = time.time()
                        proactive_elapsed_sec = 0.0
                        proactive_compute_elapsed_sec = 0.0
                        current_tree_is_proactive = False
                        if use_proactive_tree and proactive_tree is not None:
                            draft_ids = proactive_tree["draft_ids"]
                            draft_pos = proactive_tree["draft_pos"]
                            tree_mask = proactive_tree["tree_mask"]
                            parent = proactive_tree["parent"]
                            tree_depth = proactive_tree["tree_depth"]
                            final_nnodes = proactive_tree["final_nnodes"]
                            depth_widths = proactive_tree["depth_widths"]
                            node_meta = proactive_tree.get("node_meta")
                            # proactive tree expected/scale
                            runner.last_sum_expected_accepted_length = proactive_tree.get("expected_accept_length")
                            runner.last_accept_length_scale_used = float(
                                proactive_tree.get("accept_length_scale_used", 1.0)
                            )
                            use_proactive_tree = False
                            proactive_tree = None
                            current_tree_is_proactive = True
                        else:
                            # ( next_token )
                            # input_ids_t ( timeslot accepted tokens ) + next_token
                            draft_ids, draft_pos, tree_mask, parent, tree_depth, final_nnodes, depth_widths, node_meta = build_tree_with_next_token(
                                runner, input_ids_t, nodes, max_depth, current_next_token, tokenizer, debug, print_tree, per_token_probability_bound=per_token_probability_bound, per_path_probability_bound=per_path_probability_bound, min_width=min_width, fixed_width=fixed_width, fixed_width_value=fixed_width_value, fixed_nnodes=fixed_nnodes, fixed_depth=fixed_depth
                            )
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        tree_build_end_time = time.time()
                        tree_build_time = (tree_build_end_time - tree_build_start_time) * 1000.0  # ms
                        tree_build_wall_time = float(tree_build_time)
                        proactive_cancel_tree_build_overlap_ms = _record_cancel_tree_build_overlap(
                            tree_build_start_time,
                            tree_build_end_time,
                        )
                        # width
                        avg_width = float(sum(depth_widths) / len(depth_widths)) if depth_widths else 0.0
                        tree_width_records.append({
                            "question_id": question["question_id"],
                            "choice_idx": i,
                            "step": turn_steps,
                            "depth_widths": [int(w) for w in depth_widths],
                            "avg_width": avg_width,
                            "final_nnodes": int(final_nnodes),
                            "tree_depth": int(tree_depth),
                        })
                        all_final_nnodes.append(final_nnodes)
                        
                        # target
                        payload_build_start = time.perf_counter()
                        draft_input_ids_list = draft_ids[0].tolist()
                        draft_position_ids_list = draft_pos.tolist()
                        tree_mask_list = tree_mask.tolist()
                        parent_list = parent.tolist() if isinstance(parent, torch.Tensor) else list(parent)
                        payload = {
                            "type": "tree_step",
                            "draft_input_ids": draft_input_ids_list,
                            "draft_position_ids": draft_position_ids_list,
                            "tree_attention_mask": tree_mask_list,
                            "parent": parent_list,
                        }
                        payload_build_ms = max(0.0, (time.perf_counter() - payload_build_start) * 1000.0)
                        # Draft Target : Draft
                        draft_send_start_time = time.time()
                        send_json_start = time.perf_counter()
                        d2t_bytes = send_json_with_size(sock, payload)
                        send_json_ms = max(0.0, (time.perf_counter() - send_json_start) * 1000.0)
                        recv_submit_start = time.perf_counter()
                        recv_thread, recv_queue = _start_recv_json_async(sock, recv_worker=recv_worker)
                        recv_submit_ms = max(0.0, (time.perf_counter() - recv_submit_start) * 1000.0)

                        # proactive drafting: 
                        _harvest_canceled_proactive_threads(force_join=False)
                        pending_proactive = None
                        proactive_thread = None
                        proactive_stop_flag = None
                        proactive_continue_event = None
                        proactive_result = {
                            "tree": None,
                            "error": None,
                            "elapsed_sec": None,
                            "gpu_stats": None,
                            "gpu_energy_joules": 0.0,
                            "gpu_energy_kwh": 0.0,
                            "gpu_power_avg_w": 0.0,
                            "head_token": None,
                            "head_ready": False,
                            "head_ready_ts": None,
                            "final_ready": False,
                            "final_ready_ts": None,
                            "thread_exited_ts": None,
                            "cancel_set_ts": None,
                            "expected_path_tokens": None,
                            "path_match": None,
                            "budget_wait_ms": 0.0,
                            "compute_elapsed_sec": 0.0,
                            "path_prob": None,
                            "expand_decision": None,
                            "expected_gain_ms": 0.0,
                            "expected_loss_ms": 0.0,
                            "finalize_early_count": 0,
                            "expand_continue_count": 0,
                            "expand_pause_count": 0,
                            "expand_depth_counts": {},
                        }
                        proactive_path_select_ms = 0.0
                        if proactive_drafting:
                            proactive_stop_flag = threading.Event()
                            proactive_continue_event = threading.Event()
                            try:
                                path_select_start = time.perf_counter()
                                node_tokens = draft_input_ids_list[1:]  # head
                                proactive_path, proactive_path_prob = _select_proactive_path(node_tokens, parent_list, node_meta)
                                proactive_path_select_ms = max(0.0, (time.perf_counter() - path_select_start) * 1000.0)
                                should_start_proactive = False
                                if proactive_path and proactive_path_prob is not None:
                                    path_prob = float(proactive_path_prob)
                                    has_adaptive_stats = (
                                        last_target_verification_ms is not None
                                        and last_draft_to_target_ms is not None
                                        and last_target_to_draft_ms is not None
                                    )
                                    if adaptive_proactive_threshold and has_adaptive_stats:
                                        overlap_ms = min(
                                            float(tree_build_time),
                                            float(last_target_verification_ms)
                                            + float(last_draft_to_target_ms)
                                            + float(last_target_to_draft_ms),
                                        )
                                        overlap_sec = max(0.0, overlap_ms / 1000.0)
                                        expected_latency_gain_sec = path_prob * overlap_sec
                                        cancel_loss_sec = _estimate_proactive_cancel_loss_sec()
                                        expected_cancel_loss_sec = (1.0 - path_prob) * cancel_loss_sec
                                        cost_per_sec = runner.get_draft_objective_rate_per_sec()
                                        expected_cost_loss = (1.0 - path_prob) * (overlap_sec * cost_per_sec)
                                        # post-reply wait is excluded: it is normal tree-build work after use is confirmed.
                                        # Start gating only charges direct cancel tail risk plus the existing objective cost.
                                        alpha = runner.get_sensitivity_alpha()
                                        latency_gain_norm = (
                                            (expected_latency_gain_sec - expected_cancel_loss_sec)
                                            / max(1e-9, runner.get_reference_latency_per_token())
                                        )
                                        cost_loss_norm = expected_cost_loss / runner.get_reference_objective_per_token()
                                        start_value_score = ((1.0 - alpha) * latency_gain_norm - alpha * cost_loss_norm)
                                        proactive_start_expected_gain_ms_sum += expected_latency_gain_sec * 1000.0
                                        proactive_start_expected_cancel_loss_ms_sum += expected_cancel_loss_sec * 1000.0
                                        proactive_start_value_score_sum += float(start_value_score)
                                        proactive_start_value_count += 1
                                        proactive_start_cancel_loss_estimate_ms_latest = cancel_loss_sec * 1000.0
                                        if start_value_score >= 0:
                                            should_start_proactive = True
                                        else:
                                            proactive_start_skipped_by_value_count += 1
                                    else:
                                        if path_prob >= float(proactive_threshold):
                                            should_start_proactive = True

                                if should_start_proactive:
                                    proactive_result["path_prob"] = float(proactive_path_prob)
                                    proactive_result["source_step"] = int(turn_steps)
                                    # current_next_token proactive tree
                                    proactive_path = [int(current_next_token)] + [int(tok) for tok in proactive_path]
                                    proactive_result["expected_path_tokens"] = list(proactive_path)
                                    proactive_budget_sec = (
                                        float(proactive_budget_ms_latest) / 1000.0
                                        if proactive_budget_ms_latest is not None and not disable_proactive_budget
                                        else None
                                    )

                                    def _build_proactive_tree():
                                        start_time = time.time()
                                        proactive_result["started_ts"] = start_time
                                        try:
                                            tree_obj = build_proactive_tree_from_path(
                                                runner=runner,
                                                base_input_ids=input_ids_t,
                                                path_tokens=proactive_path,
                                                nodes=nodes,
                                                max_depth=max_depth,
                                                tokenizer=tokenizer,
                                                debug=debug,
                                                print_tree=False,
                                                per_token_probability_bound=per_token_probability_bound,
                                                per_path_probability_bound=per_path_probability_bound,
                                                min_width=min_width,
                                                fixed_width=fixed_width,
                                                fixed_width_value=fixed_width_value,
                                                fixed_nnodes=fixed_nnodes,
                                                fixed_depth=fixed_depth,
                                                stop_flag=proactive_stop_flag,
                                                head_token_holder=proactive_result,
                                                proactive_time_budget_sec=proactive_budget_sec,
                                                proactive_continue_event=proactive_continue_event,
                                                proactive_use_probability=proactive_path_prob,
                                                proactive_depth_stats=proactive_depth_stats,
                                                proactive_disable_budget=disable_proactive_budget,
                                            )
                                            proactive_result["tree"] = tree_obj
                                            if tree_obj is not None:
                                                timing_breakdown = tree_obj.get("timing_breakdown", {}) if isinstance(tree_obj, dict) else {}
                                                proactive_result["budget_wait_ms"] = float(
                                                    timing_breakdown.get("tree_budget_wait_ms", 0.0) or 0.0
                                                )
                                                proactive_result["expand_decision"] = timing_breakdown.get("proactive_last_expand_decision")
                                                proactive_result["expected_gain_ms"] = float(
                                                    timing_breakdown.get("proactive_expected_gain_ms", 0.0) or 0.0
                                                )
                                                proactive_result["expected_loss_ms"] = float(
                                                    timing_breakdown.get("proactive_expected_loss_ms", 0.0) or 0.0
                                                )
                                                proactive_result["finalize_early_count"] = int(
                                                    timing_breakdown.get("proactive_finalize_early_count", 0) or 0
                                                )
                                                proactive_result["expand_continue_count"] = int(
                                                    timing_breakdown.get("proactive_expand_continue_count", 0) or 0
                                                )
                                                proactive_result["expand_pause_count"] = int(
                                                    timing_breakdown.get("proactive_expand_pause_count", 0) or 0
                                                )
                                                proactive_result["expand_depth_counts"] = dict(
                                                    timing_breakdown.get("proactive_expand_depth_counts", {}) or {}
                                                )
                                                proactive_result["final_ready"] = True
                                                proactive_result["final_ready_ts"] = time.time()
                                        except Exception as e:
                                            proactive_result["error"] = e
                                        finally:
                                            end_time = time.time()
                                            proactive_result["elapsed_sec"] = end_time - start_time
                                            proactive_result["thread_exited_ts"] = end_time
                                            if runner.gpu_monitor is not None and gpu_monitor_long_running:
                                                try:
                                                    proactive_gpu_stats = runner.gpu_monitor.get_stats_between(
                                                        start_time,
                                                        end_time,
                                                    )
                                                    proactive_result["gpu_stats"] = proactive_gpu_stats
                                                    total_power_w = 0.0
                                                    if isinstance(proactive_gpu_stats, dict):
                                                        for gpu_entry in proactive_gpu_stats.values():
                                                            power_info = (
                                                                gpu_entry.get("power_draw_w", {})
                                                                if isinstance(gpu_entry, dict) else {}
                                                            )
                                                            power_avg_w = (
                                                                power_info.get("avg")
                                                                if isinstance(power_info, dict) else None
                                                            )
                                                            try:
                                                                power_avg_w = float(power_avg_w)
                                                            except Exception:
                                                                power_avg_w = 0.0
                                                            if power_avg_w > 0:
                                                                total_power_w += power_avg_w
                                                    compute_sec = max(
                                                        0.0,
                                                        float(proactive_result.get("elapsed_sec", 0.0) or 0.0)
                                                        - max(
                                                            0.0,
                                                            float(proactive_result.get("budget_wait_ms", 0.0) or 0.0)
                                                            / 1000.0,
                                                        ),
                                                    )
                                                    proactive_result["gpu_power_avg_w"] = float(total_power_w)
                                                    proactive_result["gpu_energy_joules"] = float(total_power_w * compute_sec)
                                                    proactive_result["gpu_energy_kwh"] = float(
                                                        (total_power_w * compute_sec) / 3600000.0
                                                    )
                                                except Exception as e:
                                                    proactive_result["gpu_stats_error"] = str(e)

                                    if proactive_worker is not None:
                                        proactive_thread = proactive_worker.submit(_build_proactive_tree)
                                    else:
                                        proactive_thread = threading.Thread(target=_build_proactive_tree, daemon=True)
                                        proactive_thread.start()
                                    proactive_draft_attempts += 1
                            except Exception as e:
                                if debug:
                                    print(f"[DRAFT-DEBUG] proactive drafting failed: {e}")
                        
                        # GPU (send_json_with_size() )
                        monitor_stop_start = time.perf_counter()
                        gpu_monitor_interval_end = time.time()
                        gpu_stats = None
                        if runner.gpu_monitor:
                            if gpu_monitor_long_running:
                                gpu_stats = runner.gpu_monitor.get_stats_between(
                                    gpu_monitor_interval_start,
                                    gpu_monitor_interval_end,
                                )
                            else:
                                runner.gpu_monitor.stop_monitoring()
                                gpu_stats = runner.gpu_monitor.get_stats()
                        # CPU
                        cpu_power_stats = None
                        if runner.cpu_power_monitor:
                            if debug:
                                print(f"[draft] Before stopping CPU power monitor: monitor_call_count={runner.cpu_power_monitor.monitor_call_count}, data_count={len(runner.cpu_power_monitor.data)}")
                            runner.cpu_power_monitor.stop_monitoring()
                            cpu_power_stats = runner.cpu_power_monitor.get_stats()
                            if debug:
                                print(f"[draft] CPU power statistics collected: {cpu_power_stats}")
                        else:
                            if debug:
                                print(f"[draft] warning: CPU power monitor is None (check whether enable_cpu_monitor is True)")
                        monitor_stop_ms = max(0.0, (time.perf_counter() - monitor_stop_start) * 1000.0)

                        # Target Draft recv .
                        recv_wait_start = time.perf_counter()
                        reply, t2d_bytes, draft_recv_end_time = _await_recv_json_async(
                            recv_thread, recv_queue
                        )
                        recv_wait_ms = max(0.0, (time.perf_counter() - recv_wait_start) * 1000.0)
                        
                        # Per-question transfer byte counters
                        draft_to_target_bytes_per_question.append(d2t_bytes)
                        target_to_draft_bytes_per_question.append(t2d_bytes)
                        run_draft_to_target_bytes.append(d2t_bytes)
                        run_target_to_draft_bytes.append(t2d_bytes)
                        assert reply.get("type") == "verify_result"
                        accepted_tokens: List[int] = reply["accepted_tokens"]
                        accept_length: int = reply["accept_length"]
                        next_token: int = reply["next_token"]
                        eos_reached: bool = reply["eos_reached"]
                        best_ids: List[int] = reply["best_ids"]
                        target_energy_rate_per_sec = reply.get("target_energy_rate_per_sec", None)

                        # proactive tree
                        proactive_post_reply_wait_ms = 0.0
                        proactive_budget_wait_ms = 0.0
                        proactive_resume_after_reply_ms = 0.0
                        proactive_cancel_immediate_join_ms = 0.0
                        proactive_compute_hidden_before_reply_sec = 0.0
                        proactive_compute_after_reply_sec = 0.0
                        proactive_head_ready_at_reply = False
                        proactive_final_ready_at_reply = False
                        proactive_cancel_reason = None
                        proactive_path_match = None
                        proactive_shape_at_post_reply = None
                        proactive_decision_start = time.perf_counter()
                        if proactive_drafting:
                            canceled = False
                            if proactive_thread is not None:
                                proactive_head = proactive_result.get("head_token")
                                proactive_head_ready_at_reply = bool(proactive_result.get("head_ready", False)) or (
                                    proactive_head is not None
                                )
                                proactive_final_ready_at_reply = bool(proactive_result.get("final_ready", False))
                                proactive_cancel_reason = None
                                # Cheap checks first; path comparison only runs when depth/head already match.
                                if accept_length != tree_depth:
                                    proactive_cancel_reason = "accept_length_mismatch"
                                elif proactive_head is None or next_token != proactive_head:
                                    proactive_cancel_reason = "head_mismatch"
                                else:
                                    expected_path_tokens = proactive_result.get("expected_path_tokens") or []
                                    if expected_path_tokens:
                                        try:
                                            accepted_path_tokens = [int(tok) for tok in accepted_tokens]
                                            expected_path_tokens = [int(tok) for tok in expected_path_tokens]
                                            proactive_path_match = accepted_path_tokens == expected_path_tokens
                                        except Exception:
                                            proactive_path_match = False
                                    else:
                                        proactive_path_match = False
                                    proactive_result["path_match"] = proactive_path_match
                                    if proactive_path_match:
                                        proactive_path_match_count += 1
                                    else:
                                        proactive_path_mismatch_count += 1
                                        proactive_cancel_reason = "path_mismatch"
                                should_use_proactive = proactive_cancel_reason is None
                                if not should_use_proactive:
                                    canceled = True
                                    proactive_stop_flag.set()
                                    proactive_result["cancel_set_ts"] = time.time()
                                    runner.reset_proactive_kv()
                                    pending_proactive = None
                                    _pending_canceled_proactive.append(
                                        {
                                            "thread": proactive_thread,
                                            "state": proactive_result,
                                            "cancel_ts": proactive_result.get("cancel_set_ts"),
                                            "reason": proactive_cancel_reason,
                                        }
                                    )
                                    proactive_cancel_immediate_join_ms = _join_canceled_proactive_immediately()
                                    proactive_elapsed_sec = float(
                                        proactive_result.get("elapsed_sec", 0.0) or 0.0
                                    )
                                    proactive_compute_elapsed_sec = float(
                                        proactive_result.get("compute_elapsed_sec", 0.0) or 0.0
                                    )
                                else:
                                    resume_signal_start_ts = time.time()
                                    if proactive_continue_event is not None:
                                        proactive_continue_event.set()
                                    proactive_resume_after_reply_ms = max(
                                        0.0, (time.time() - resume_signal_start_ts) * 1000.0
                                    )
                                    proactive_resume_after_reply_ms_sum += float(proactive_resume_after_reply_ms)
                                    proactive_resume_after_reply_ms_count += 1
                                    proactive_resume_after_reply_ms_max = max(
                                        proactive_resume_after_reply_ms_max,
                                        float(proactive_resume_after_reply_ms),
                                    )
                                    if proactive_final_ready_at_reply:
                                        pending_proactive = proactive_result["tree"]
                                    else:
                                        wait_start_ts = time.time()
                                        proactive_thread.join()
                                        proactive_post_reply_wait_ms = max(
                                            0.0, (time.time() - wait_start_ts) * 1000.0
                                        )
                                        proactive_post_reply_wait_ms_sum += float(proactive_post_reply_wait_ms)
                                        proactive_post_reply_wait_ms_count += 1
                                        proactive_post_reply_wait_ms_max = max(
                                            proactive_post_reply_wait_ms_max,
                                            float(proactive_post_reply_wait_ms),
                                        )
                                    if proactive_result["error"] and debug:
                                        print(f"[DRAFT-DEBUG] proactive drafting failed: {proactive_result['error']}")
                                    if pending_proactive is None:
                                        pending_proactive = proactive_result["tree"]
                                    proactive_shape_at_post_reply = _record_proactive_post_reply_shape(
                                        pending_proactive,
                                        proactive_post_reply_wait_ms,
                                    )
                                    proactive_budget_wait_ms = float(
                                        proactive_result.get("budget_wait_ms", 0.0) or 0.0
                                    )
                                    if proactive_budget_wait_ms > 0:
                                        proactive_budget_wait_ms_sum += proactive_budget_wait_ms
                                        proactive_budget_wait_ms_count += 1
                                        proactive_budget_wait_ms_max = max(
                                            proactive_budget_wait_ms_max,
                                            proactive_budget_wait_ms,
                                        )
                                    if proactive_result.get("elapsed_sec") is not None:
                                        proactive_elapsed_sec = float(proactive_result["elapsed_sec"])
                                        proactive_compute_elapsed_sec = _account_proactive_compute(
                                            proactive_result,
                                            used=True,
                                            post_reply_wait_ms=proactive_post_reply_wait_ms,
                                        )
                                        proactive_compute_hidden_before_reply_sec = float(
                                            proactive_result.get("compute_hidden_before_reply_sec", 0.0) or 0.0
                                        )
                                        proactive_compute_after_reply_sec = float(
                                            proactive_result.get("compute_after_reply_sec", 0.0) or 0.0
                                        )

                            if (
                                pending_proactive
                                and accept_length == tree_depth
                                and next_token == pending_proactive.get("head_token")
                                and proactive_path_match is True
                            ):
                                _merge_proactive_expand_metrics(proactive_result)
                                _record_proactive_depth_outcome(pending_proactive.get("tree_depth", tree_depth), "used")
                                use_proactive_tree = True
                                proactive_tree = pending_proactive
                                # proactive KV KV
                                runner.draft_stable_kv = runner.proactive_kv
                                proactive_tree_used += 1
                                if proactive_path_prob is not None:
                                    proactive_used_path_prob_sum += float(proactive_path_prob)
                                    proactive_used_path_prob_count += 1
                            else:
                                if proactive_thread is not None:
                                    _merge_proactive_expand_metrics(proactive_result)
                                    failed_depth = (
                                        pending_proactive.get("tree_depth", tree_depth)
                                        if isinstance(pending_proactive, dict)
                                        else tree_depth
                                    )
                                    _record_proactive_depth_outcome(failed_depth, "canceled")
                                if not canceled:
                                    if proactive_thread is not None:
                                        proactive_stop_flag.set()
                                    runner.reset_proactive_kv()
                                use_proactive_tree = False
                                proactive_tree = None
                                if proactive_path_prob is not None:
                                    proactive_unused_path_prob_sum += float(proactive_path_prob)
                                    proactive_unused_path_prob_count += 1
                        else:
                            runner.reset_proactive_kv()
                            use_proactive_tree = False
                            proactive_tree = None
                        proactive_decision_ms = max(0.0, (time.perf_counter() - proactive_decision_start) * 1000.0)
                        # proactive compute tree_build_time . compute
                        # proactive_unused / .
                        if proactive_compute_elapsed_sec > 0 and use_proactive_tree:
                            tree_build_time += proactive_compute_elapsed_sec * 1000.0
                        
                        target_recv_end_time = reply.get("target_recv_end_time", None)
                        target_send_start_time = reply.get("target_send_start_time", None)
                        target_verification_time = reply.get("target_verification_time_ms", None)  # Target (ms)
                        
                        # Draft Target
                        draft_to_target_time = None
                        if target_recv_end_time is not None :
                            draft_to_target_time = max(0.05, (target_recv_end_time - draft_send_start_time) * 1000.0)  # ms
                            draft_to_target_time_per_question.append(draft_to_target_time)
                            # final_nnodes draft_to_target_time
                            if final_nnodes not in d2t_per_final_nnodes_per_question:
                                d2t_per_final_nnodes_per_question[final_nnodes] = []
                            d2t_per_final_nnodes_per_question[final_nnodes].append(draft_to_target_time)
                        
                        # Target Draft
                        target_to_draft_time = None
                        if target_send_start_time is not None:
                            target_to_draft_time = max(0.05, (draft_recv_end_time - target_send_start_time) * 1000.0)  # ms
                            target_to_draft_time_per_question.append(target_to_draft_time)
                            if accept_length not in t2d_per_accept_len_per_question:
                                t2d_per_accept_len_per_question[accept_length] = []
                            t2d_per_accept_len_per_question[accept_length].append(target_to_draft_time)
                        
                        # Latency
                        step_latency = {
                            "tree_build_time_ms": tree_build_time,
                            "tree_build_wall_time_ms": tree_build_wall_time,
                            "draft_to_target_time_ms": draft_to_target_time,
                            "target_verification_time_ms": target_verification_time,
                            "target_to_draft_time_ms": target_to_draft_time,
                            "payload_build_ms": payload_build_ms,
                            "send_json_ms": send_json_ms,
                            "recv_submit_ms": recv_submit_ms,
                            "recv_wait_ms": recv_wait_ms,
                            "proactive_path_select_ms": proactive_path_select_ms,
                            "proactive_decision_ms": proactive_decision_ms,
                            "monitor_stop_ms": monitor_stop_ms,
                            "tree_model_forward_ms": float(runner.last_tree_timing_breakdown.get("tree_model_forward_ms", 0.0) or 0.0),
                            "tree_width_algo_ms": float(runner.last_tree_timing_breakdown.get("tree_width_algo_ms", 0.0) or 0.0),
                            "tree_nnodes_algo_ms": float(runner.last_tree_timing_breakdown.get("tree_nnodes_algo_ms", 0.0) or 0.0),
                            "tree_mask_build_ms": float(runner.last_tree_timing_breakdown.get("tree_mask_build_ms", 0.0) or 0.0),
                            "tree_finalize_ms": float(runner.last_tree_timing_breakdown.get("tree_finalize_ms", 0.0) or 0.0),
                            "tree_budget_wait_ms": float(runner.last_tree_timing_breakdown.get("tree_budget_wait_ms", 0.0) or 0.0),
                            "draft_to_target_bytes": d2t_bytes,
                            "target_to_draft_bytes": t2d_bytes,
                            "accept_length": accept_length,
                            "current_tree_is_proactive": bool(current_tree_is_proactive),
                            "previous_step_used_proactive_tree": bool(prev_step_used_proactive_tree),
                            "previous_step_proactive_canceled": bool(prev_step_proactive_canceled),
                            # Draft expected accept length (tree.sum_expected_accepted_length)
                            "expected_accept_length": runner.last_sum_expected_accepted_length,
                            "proactive_wall_elapsed_ms": float(proactive_elapsed_sec * 1000.0),
                            "proactive_compute_elapsed_ms": float(proactive_compute_elapsed_sec * 1000.0),
                            "proactive_compute_hidden_before_reply_ms": float(
                                proactive_compute_hidden_before_reply_sec * 1000.0
                            ),
                            "proactive_compute_after_reply_ms": float(
                                proactive_compute_after_reply_sec * 1000.0
                            ),
                            "proactive_post_reply_wait_ms": proactive_post_reply_wait_ms,
                            "proactive_budget_ms": proactive_budget_ms_latest,
                            "proactive_budget_wait_ms": proactive_budget_wait_ms,
                            "proactive_resume_after_reply_ms": proactive_resume_after_reply_ms,
                            "proactive_cancel_immediate_join_ms": proactive_cancel_immediate_join_ms,
                            "proactive_cancel_join_before_tree_build_ms": proactive_cancel_join_before_tree_build_ms,
                            "proactive_cancel_tree_build_overlap_ms": proactive_cancel_tree_build_overlap_ms,
                            "proactive_head_ready_at_reply": proactive_head_ready_at_reply,
                            "proactive_final_ready_at_reply": proactive_final_ready_at_reply,
                            "proactive_path_match": proactive_path_match,
                            "proactive_cancel_reason": proactive_cancel_reason,
                            "proactive_post_reply_tree_depth": (
                                proactive_shape_at_post_reply.get("depth")
                                if proactive_shape_at_post_reply is not None
                                else None
                            ),
                            "proactive_post_reply_avg_width": (
                                proactive_shape_at_post_reply.get("avg_width")
                                if proactive_shape_at_post_reply is not None
                                else None
                            ),
                            "proactive_post_reply_width_bin": (
                                proactive_shape_at_post_reply.get("width_bin")
                                if proactive_shape_at_post_reply is not None
                                else None
                            ),
                            "proactive_post_reply_final_nnodes": (
                                proactive_shape_at_post_reply.get("final_nnodes")
                                if proactive_shape_at_post_reply is not None
                                else None
                            ),
                            "proactive_start_path_prob": (
                                float(proactive_result.get("path_prob"))
                                if proactive_result.get("path_prob") is not None
                                else None
                            ),
                            "proactive_expand_decision": proactive_result.get("expand_decision"),
                            "proactive_expected_gain_ms": float(proactive_result.get("expected_gain_ms", 0.0) or 0.0),
                            "proactive_expected_loss_ms": float(proactive_result.get("expected_loss_ms", 0.0) or 0.0),
                            "proactive_finalize_early_count": int(proactive_result.get("finalize_early_count", 0) or 0),
                            "proactive_expand_continue_count": int(proactive_result.get("expand_continue_count", 0) or 0),
                            "proactive_expand_pause_count": int(proactive_result.get("expand_pause_count", 0) or 0),
                            "proactive_start_cancel_loss_estimate_ms": (
                                float(proactive_start_cancel_loss_estimate_ms_latest)
                                if proactive_start_cancel_loss_estimate_ms_latest is not None
                                else None
                            ),
                        }
                        latency_data.append(step_latency)
                        stats_update_start = time.perf_counter()
                        if tree_build_time is not None:
                            last_tree_build_ms = float(tree_build_time)
                        if target_verification_time is not None:
                            last_target_verification_ms = float(target_verification_time)
                            tvals = target_verify_per_nnodes_per_question.setdefault(int(final_nnodes), [])
                            tvals.append(float(target_verification_time))
                        if draft_to_target_time is not None:
                            last_draft_to_target_ms = float(draft_to_target_time)
                        if target_to_draft_time is not None:
                            last_target_to_draft_ms = float(target_to_draft_time)
                        _refresh_proactive_budget()

                        # accept_length ( scale=actual_sum/expected_sum )
                        if bool(accept_length_calibration):
                            try:
                                exp_len = float(runner.last_sum_expected_accepted_length) if runner.last_sum_expected_accepted_length is not None else None
                            except Exception:
                                exp_len = None
                            if exp_len is not None and exp_len > 0 and accept_length is not None and int(accept_length) > 0:
                                runner.accept_length_actual_sum += float(int(accept_length))
                                runner.accept_length_expected_sum += float(exp_len)
                        if current_tree_is_proactive:
                            proactive_used_tree_steps += 1
                            proactive_used_tree_depth_sum += float(tree_depth)
                            proactive_used_tree_width_sum += float(avg_width)
                            proactive_used_tree_final_nodes_sum += float(final_nnodes)
                            proactive_used_tree_accept_length_sum += float(accept_length)
                            if runner.last_sum_expected_accepted_length is not None:
                                try:
                                    proactive_used_tree_scaled_expected_sum += (
                                        float(runner.last_sum_expected_accepted_length)
                                        * float(getattr(runner, "last_accept_length_scale_used", 1.0))
                                    )
                                except Exception:
                                    pass
                        else:
                            foreground_tree_steps += 1
                            foreground_tree_accept_length_sum += float(accept_length)
                            if prev_step_used_proactive_tree:
                                after_proactive_foreground_steps += 1
                                after_proactive_foreground_accept_length_sum += float(accept_length)
                            if prev_step_proactive_canceled:
                                after_proactive_cancel_foreground_steps += 1
                                after_proactive_cancel_foreground_accept_length_sum += float(accept_length)

                        # target_verification_time / target_profile_data 
                        # key Tree.get_objective_value nnodes_{nnodes} .
                        if bool(target_time_calibration) and target_verification_time is not None and runner.target_profile_data is not None:
                            try:
                                profile_key = f"nnodes_{int(final_nnodes)}"
                                predicted_ms = runner.target_profile_data.get(profile_key, {}).get("avg_time_ms", None)
                                if predicted_ms is not None and predicted_ms > 0:
                                    ratio = float(target_verification_time) / float(predicted_ms)
                                    if ratio == ratio and ratio > 0:
                                        runner.target_verification_ratio_sum += float(ratio)
                                        runner.target_verification_ratio_count += 1
                            except Exception:
                                pass
                        
                        # GPU 
                        if gpu_stats:
                            step_time = tree_build_time / 1000.0  # ms
                            runner.update_draft_objective_rate_from_gpu(
                                gpu_stats,
                                require_valid=bool(runner.uses_total_cost_objective() and not runner.no_draft_cost),
                            )
                            draft_gpu_data.append({
                                "step": turn_steps,
                                "timestamp": datetime.now().isoformat(),
                                "gpu_stats": gpu_stats,
                                "monitor_call_count": runner.gpu_monitor.monitor_call_count if runner.gpu_monitor else None
                            })
                            draft_timing_data.append({
                                "total_time_seconds": step_time,
                                "timestamp": datetime.now().isoformat(),
                            })
                        runner.update_target_objective_rate(target_energy_rate_per_sec)
                        # CPU
                        if cpu_power_stats:
                            step_time = tree_build_time / 1000.0  # ms
                            draft_cpu_power_data.append({
                                "step": turn_steps,
                                "timestamp": datetime.now().isoformat(),
                                "cpu_power_stats": cpu_power_stats,
                                "monitor_call_count": runner.cpu_power_monitor.monitor_call_count if runner.cpu_power_monitor else None
                            })
                            if debug:
                                print(f"[draft] CPU power data added (step: {turn_steps}, total collected count: {len(draft_cpu_power_data)})")
                        else:
                            if debug:
                                print(f"[draft] warning: CPU power statistics are None (step: {turn_steps})")
                        step_target_energy_kwh, step_target_energy_measured = _target_energy_kwh_from_step(
                            target_verification_time,
                            target_energy_rate_per_sec,
                        )
                        total_target_energy_kwh += float(step_target_energy_kwh)
                        if step_target_energy_measured:
                            target_energy_sample_count += 1
                        else:
                            target_energy_missing_count += 1
                        
                        if debug:
                            print(f"[DRAFT-DEBUG] Draft -> Target transfer time: {draft_to_target_time:.3f}ms")
                            print(f"[DRAFT-DEBUG] Target -> Draft transfer time: {target_to_draft_time:.3f}ms")
                            print(f"[DRAFT-DEBUG] Draft -> Target transfer size: {d2t_bytes}bytes")
                            print(f"[DRAFT-DEBUG] Target -> Draft transfer size: {t2d_bytes}bytes")
                        step_latency["stats_update_ms"] = max(
                            0.0, (time.perf_counter() - stats_update_start) * 1000.0
                        )
                        
                        # tree ( tree )
                        # timing_stats total_time_seconds ,
                        timing_stats = reply.get("timing_stats", {})
                        # total_time = timing_stats.get("total_time_seconds", None)
                        # if total_time is None:
                        # (Draft Target Draft)
                        total_time = max(0.0, target_recv_end_time - draft_send_start_time)
                        
                        transfer_time = None
                        if draft_to_target_time is not None and target_to_draft_time is not None:
                            transfer_time = (draft_to_target_time + target_to_draft_time) / 1000.0  # ms
                        
                        runner.prev_tree_final_nnodes = final_nnodes
                        runner.prev_tree_depth = tree_depth
                        runner.prev_tree_total_target_time = total_time
                        runner.prev_tree_transfer_time = transfer_time
                        runner.prev_tree_accept_length = accept_length
                        if draft_to_target_time is not None and final_nnodes > 0:
                            runner.per_token_draft_to_target_transfer_time = (draft_to_target_time / 1000.0) / final_nnodes
                            runner.per_token_draft_to_target_bytes = float(d2t_bytes) / final_nnodes
                        if target_to_draft_time is not None and accept_length > 0:
                            runner.per_token_target_to_draft_transfer_time = (target_to_draft_time / 1000.0) / accept_length
                            runner.per_token_target_to_draft_bytes = float(t2d_bytes) / accept_length

                        # TODO:Accept KV
                        # base_kv_len = input_ids_t.shape[1]
                        # update_kv_with_accepted(runner, best_ids, base_kv_len, next_token, debug)

                        input_update_start = time.perf_counter()
                        # . input_ids_t + accepted_tokens
                        input_ids_t = torch.cat([
                            input_ids_t,
                            torch.tensor([accepted_tokens], device=input_ids_t.device, dtype=torch.long),
                        ], dim=-1)

                        output_tokens.extend(accepted_tokens)
                        new_token_count += accept_length + 1
                        turn_steps += 1

                        grand_total_steps += 1
                        grand_total_accepted += accept_length
                        grand_total_draft_tokens += int(tree_depth)
                        # accept_length
                        # accept_length_list.append(int(accept_length))
                        accept_length_per_question.append(int(accept_length))
                        all_accept_lengths.append(int(accept_length))  # accept_stats
                        step_latency["input_update_ms"] = max(
                            0.0, (time.perf_counter() - input_update_start) * 1000.0
                        )
                        step_metric = _estimate_step_metric_spend(
                            tree_build_time_ms=tree_build_time,
                            target_verification_time_ms=target_verification_time,
                            d2t_step_bytes=d2t_bytes,
                            t2d_step_bytes=t2d_bytes,
                            gpu_stats_step=gpu_stats,
                            target_energy_rate_per_sec=target_energy_rate_per_sec,
                            server_only_step=False,
                        )
                        step_total_time_ms = (
                            float(tree_build_time or 0.0)
                            + float(draft_to_target_time or 0.0)
                            + float(target_verification_time or 0.0)
                            + float(target_to_draft_time or 0.0)
                        )
                        step_tokens_for_ref = float(int(accept_length) + 1)
                        if step_total_time_ms > 0 and step_tokens_for_ref > 0:
                            ref_update_total_time_sec += float(step_total_time_ms / 1000.0)
                            ref_update_total_tokens += float(step_tokens_for_ref)
                            ref_update_total_metric += float(max(0.0, float(step_metric)))
                        metric_spent_total += float(step_metric)
                        if np.isfinite(total_metric_cap) and metric_spent_total >= float(total_metric_cap):
                            metric_cap_reached = True
                            if debug:
                                print(
                                    f"[DRAFT-DEBUG] metric cap reached: "
                                    f"spent={metric_spent_total:.6f}, cap={float(total_metric_cap):.6f}"
                                )
                            break
                        exp_raw = None
                        exp_scaled = None
                        exp_clipped = None
                        try:
                            exp_raw_candidate = (
                                float(runner.last_sum_expected_accepted_length)
                                if runner.last_sum_expected_accepted_length is not None
                                else None
                            )
                            if exp_raw_candidate is not None and np.isfinite(exp_raw_candidate):
                                exp_raw = exp_raw_candidate
                                scale_used = float(getattr(runner, "last_accept_length_scale_used", 1.0))
                                if np.isfinite(scale_used):
                                    exp_scaled = exp_raw * scale_used
                                    exp_clipped = exp_scaled * (1.0 - float(getattr(runner, "accept_length_margin", 0.05)))
                        except Exception:
                            exp_raw = None
                            exp_scaled = None
                            exp_clipped = None
                        all_expected_accept_lengths_per_step.append(exp_raw)
                        all_scaled_expected_accept_lengths_per_step.append(exp_scaled)
                        all_clipped_expected_accept_lengths_per_step.append(exp_clipped)
                        accept_length_step_pairs["actual_accept_length_per_step"].append(int(accept_length))
                        accept_length_step_pairs["expected_accept_length_raw_per_step"].append(exp_raw)
                        accept_length_step_pairs["expected_accept_length_scaled_per_step"].append(exp_scaled)
                        accept_length_step_pairs["expected_accept_length_clipped_per_step"].append(exp_clipped)

                        # next_token
                        current_next_token = next_token
                        prev_step_used_proactive_tree = bool(current_tree_is_proactive)
                        prev_step_proactive_canceled = bool(proactive_cancel_reason is not None)

                        if eos_reached or new_token_count >= int(max_new_tokens) or input_ids_t.shape[1] > 1960:
                            break
                    
                    torch.cuda.synchronize()
                    turn_end_time = time.time()
                    turn_wall_time = turn_end_time - turn_start_time
                    if server_only_mode:
                        # server-only turn_wall_time
                        server_only_total_session_time_sec += float(turn_wall_time)
                    
                    # turn width draft model call query .
                    for _w, _times in getattr(runner, "question_width_times", {}).items():
                        try:
                            w_key = int(_w)
                        except Exception:
                            continue
                        if not isinstance(_times, list) or (not _times):
                            continue
                        dst = draft_model_call_ms_per_width_per_question.setdefault(w_key, [])
                        for _sec in _times:
                            try:
                                ms_val = float(_sec) * 1000.0
                                if np.isfinite(ms_val) and ms_val >= 0:
                                    dst.append(ms_val)
                            except Exception:
                                continue

                    # width draft_model.model
                    # runner.print_timing_stats()
                    runner.reset_timing_stats()

                    # Stop token 
                    if conv.stop_token_ids:
                        stop_token_ids_index = [
                            idx
                            for idx, token_id in enumerate(output_tokens)
                            if token_id in conv.stop_token_ids
                        ]
                        if len(stop_token_ids_index) > 0:
                            output_tokens = output_tokens[:stop_token_ids_index[0]]
                    
                    decoded = tokenizer.decode(output_tokens, spaces_between_special_tokens=False)
                    
                    # Stop string
                    if conv.stop_str and decoded.find(conv.stop_str) > 0:
                        decoded = decoded[:decoded.find(conv.stop_str)]
                    
                    # Special token
                    for special_token in tokenizer.special_tokens_map.values():
                        if isinstance(special_token, list):
                            for special_tok in special_token:
                                decoded = decoded.replace(special_tok, "")
                        else:
                            decoded = decoded.replace(special_token, "")
                    
                    decoded = decoded.strip()
                    
                    if conv.name == "xgen" and decoded.startswith("Assistant:"):
                        decoded = decoded.replace("Assistant:", "", 1).strip()

                    turns.append(decoded)
                    idxs.append(int(turn_steps))
                    new_tokens_list.append(int(new_token_count))
                    wall_time_list.append(turn_wall_time)
                    
                    conv.messages[-1][-1] = decoded

                avg_draft_to_target = float(sum(draft_to_target_time_per_question) / len(draft_to_target_time_per_question)) if draft_to_target_time_per_question else None
                avg_target_to_draft = float(sum(target_to_draft_time_per_question) / len(target_to_draft_time_per_question)) if target_to_draft_time_per_question else None
                avg_accept_length = float(sum(accept_length_per_question) / len(accept_length_per_question)) if accept_length_per_question else None
                accept_len_bucket_stats = {}
                for length, times in t2d_per_accept_len_per_question.items():
                    if not times:
                        continue
                    accept_len_bucket_stats[int(length)] = {
                        "count": len(times),
                        "avg_target_to_draft_time_ms": float(sum(times) / len(times)),
                        "min_target_to_draft_time_ms": float(min(times)),
                        "max_target_to_draft_time_ms": float(max(times)),
                        "times_ms": [float(t) for t in times],
                    }
                # final_nnodes draft_to_target_time
                final_nnodes_bucket_stats = {}
                for nnodes, times in d2t_per_final_nnodes_per_question.items():
                    if not times:
                        continue
                    final_nnodes_bucket_stats[int(nnodes)] = {
                        "count": len(times),
                        "avg_draft_to_target_time_ms": float(sum(times) / len(times)),
                        "min_draft_to_target_time_ms": float(min(times)),
                        "max_draft_to_target_time_ms": float(max(times)),
                        "times_ms": [float(t) for t in times],
                    }
                

                choices.append({
                    "index": i, 
                    "turns": turns, 
                    "idxs": idxs, 
                    "new_tokens": new_tokens_list, 
                    "wall_time": wall_time_list,
                })

            # Network latency and accept-length records
            network_latency_records.append({
                "question_id": question["question_id"],
                "draft_to_target_time_ms": draft_to_target_time_per_question,
                "target_to_draft_time_ms": target_to_draft_time_per_question,
                "draft_to_target_bytes": draft_to_target_bytes_per_question,
                "target_to_draft_bytes": target_to_draft_bytes_per_question,
                "accept_length": accept_length_per_question,
                "avg_draft_to_target_time_ms": avg_draft_to_target,
                "avg_target_to_draft_time_ms": avg_target_to_draft,
                "avg_accept_length": avg_accept_length,
                "accept_length_bucket_stats": accept_len_bucket_stats,
                "final_nnodes_bucket_stats": final_nnodes_bucket_stats,
            })

            global_draft_to_target_times.extend(draft_to_target_time_per_question)
            global_target_to_draft_times.extend(target_to_draft_time_per_question)
            global_accept_lengths.extend(accept_length_per_question)
            for length, times in t2d_per_accept_len_per_question.items():
                if length not in global_t2d_per_accept_len:
                    global_t2d_per_accept_len[length] = []
                global_t2d_per_accept_len[length].extend(times)
            # final_nnodes draft_to_target_time
            for nnodes, times in d2t_per_final_nnodes_per_question.items():
                if nnodes not in global_d2t_per_final_nnodes:
                    global_d2t_per_final_nnodes[nnodes] = []
                global_d2t_per_final_nnodes[nnodes].extend(times)

            online_profile_update_result = _apply_online_profile_updates_and_flush(
                runner,
                observed_width_ms=draft_model_call_ms_per_width_per_question,
                observed_nnodes_ms=target_verify_per_nnodes_per_question,
                debug=debug,
            )
            question_profile_update_overhead_sec = float(
                online_profile_update_result.get("overhead_sec", 0.0) or 0.0
            )
            profile_update_overhead_sec_total += max(0.0, question_profile_update_overhead_sec)

            # Query cumulative reference
            # ( , speculative decoding step )
            ref_tps_before_query_update = float(runner.reference_tps)
            ref_cost_before_query_update = float(runner.reference_cost_per_token)
            ref_obj_before_query_update = float(runner.reference_objective_per_token)
            if ref_update_total_time_sec > 0.0 and ref_update_total_tokens > 0.0:
                updated_ref_tps = float(ref_update_total_tokens / max(1e-9, ref_update_total_time_sec))
                updated_ref_obj = float(ref_update_total_metric / max(1e-9, ref_update_total_tokens))
                if np.isfinite(updated_ref_tps) and updated_ref_tps > 0.0:
                    runner.reference_tps = float(updated_ref_tps)
                if np.isfinite(updated_ref_obj) and updated_ref_obj >= 0.0:
                    runner.reference_objective_per_token = float(updated_ref_obj)
                    if objective_metric in cost_objective_metrics:
                        runner.reference_cost_per_token = float(updated_ref_obj)
            ref_tps_after_query_update = float(runner.reference_tps)
            ref_cost_after_query_update = float(runner.reference_cost_per_token)
            ref_obj_after_query_update = float(runner.reference_objective_per_token)

            if network_latency_records:
                _latest_q = network_latency_records[-1]
                _latest_q["reference_tps_before_query_update"] = float(ref_tps_before_query_update)
                _latest_q["reference_tps_after_query_update"] = float(ref_tps_after_query_update)
                _latest_q["reference_cost_per_token_before_query_update"] = float(ref_cost_before_query_update)
                _latest_q["reference_cost_per_token_after_query_update"] = float(ref_cost_after_query_update)
                _latest_q["reference_objective_per_token_before_query_update"] = float(ref_obj_before_query_update)
                _latest_q["reference_objective_per_token_after_query_update"] = float(ref_obj_after_query_update)
                _latest_q["reference_update_cumulative_time_seconds"] = float(ref_update_total_time_sec)
                _latest_q["reference_update_cumulative_tokens"] = float(ref_update_total_tokens)
                _latest_q["reference_update_cumulative_metric"] = float(ref_update_total_metric)

            reference_update_history.append({
                "query_index": int(len(reference_update_history)),
                "question_id": question.get("question_id"),
                "reference_tps_before": float(ref_tps_before_query_update),
                "reference_tps_after": float(ref_tps_after_query_update),
                "reference_cost_per_token_before": float(ref_cost_before_query_update),
                "reference_cost_per_token_after": float(ref_cost_after_query_update),
                "reference_objective_per_token_before": float(ref_obj_before_query_update),
                "reference_objective_per_token_after": float(ref_obj_after_query_update),
                "cumulative_time_seconds": float(ref_update_total_time_sec),
                "cumulative_tokens": float(ref_update_total_tokens),
                "cumulative_metric": float(ref_update_total_metric),
            })
            
            # all_accept_lengths ( step )
            
            question_elapsed_time = time.time() - question_start_time
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": question_elapsed_time,
                "profile_update_overhead_seconds": float(question_profile_update_overhead_sec),
            }
            all_answers.append(ans_json)

        total_time = max(
            1e-6,
            time.time() - run_start_time - float(server_only_profile_prepare_overhead_sec),
        )
        tokens_per_step = (grand_total_accepted + grand_total_steps) / max(1, grand_total_steps)  # (accept_length+1)
        tokens_per_second_wall = (grand_total_accepted + grand_total_steps) / total_time
        # Wall-time TPS primary (profile update overhead ).
        tokens_per_second = float(tokens_per_second_wall)
        avg_tree_depth = (grand_total_draft_tokens / max(1, grand_total_steps)) if grand_total_steps > 0 else 0.0
        avg_accept_length = (grand_total_accepted / max(1, grand_total_steps)) if grand_total_steps > 0 else 0.0
        reported_acceptance_ratio_avg = (
            (grand_total_accepted / max(1, grand_total_draft_tokens))
            if grand_total_draft_tokens > 0 else 0.0
        )
        if bool(force_server_only_ar) and (grand_total_accepted + grand_total_steps) > 0:
            # AR emits exactly one target token per model step. Keep internal
            # token accounting intact for TPS/cost, but report AR-specific
            # accept/step metrics in token units instead of speculative units.
            tokens_per_step = 1.0
            avg_accept_length = 1.0
            reported_acceptance_ratio_avg = 1.0
        avg_tree_width = float(sum(r["avg_width"] for r in tree_width_records) / len(tree_width_records)) if tree_width_records else 0.0
        avg_final_nodes = float(sum(r["final_nnodes"] for r in tree_width_records) / len(tree_width_records)) if tree_width_records else 0.0
        # depth width
        depth_width_sum = {}
        depth_width_count = {}
        for rec in tree_width_records:
            for depth_idx, w in enumerate(rec.get("depth_widths", []), start=1):
                depth_width_sum[depth_idx] = depth_width_sum.get(depth_idx, 0.0) + w
                depth_width_count[depth_idx] = depth_width_count.get(depth_idx, 0) + 1
        depth_avg_widths = {
            depth: float(depth_width_sum[depth] / depth_width_count[depth])
            for depth in sorted(depth_width_sum.keys())
        } if depth_width_sum else {}
        
        # Accept stats (target )
        accept_stats = {}
        if all_accept_lengths:
            accept_stats = {
                "total_accepted_tokens": sum(all_accept_lengths),
                "avg_accept_length": float(sum(all_accept_lengths) / len(all_accept_lengths)),
                "min_accept_length": int(min(all_accept_lengths)),
                "max_accept_length": int(max(all_accept_lengths)),
                "num_steps": len(all_accept_lengths)
            }
        
        # Expected accept length ( step " " sum_expected_accepted_length )
        # NOTE: runner.accumulated_depth_stats (depth ) .
        if all_expected_accept_lengths_per_step:
            valid_lengths = []
            for v in all_expected_accept_lengths_per_step:
                if v is None:
                    continue
                if isinstance(v, (int, float)) and v == v:  # NaN : v==v NaN True
                    valid_lengths.append(float(v))
            avg_expected_accept_length = float(sum(valid_lengths) / len(valid_lengths)) if valid_lengths else None
        else:
            avg_expected_accept_length = None

        # Scaled expected accept length 
        if all_scaled_expected_accept_lengths_per_step:
            valid_scaled = []
            for v in all_scaled_expected_accept_lengths_per_step:
                if v is None:
                    continue
                if isinstance(v, (int, float)) and v == v:
                    valid_scaled.append(float(v))
            avg_scaled_expected_accept_length = float(sum(valid_scaled) / len(valid_scaled)) if valid_scaled else None
        else:
            avg_scaled_expected_accept_length = None

        # Clipped expected accept length (scaled * (1-accept_length_margin))
        if all_clipped_expected_accept_lengths_per_step:
            valid_clipped = []
            for v in all_clipped_expected_accept_lengths_per_step:
                if v is None:
                    continue
                if isinstance(v, (int, float)) and v == v:
                    valid_clipped.append(float(v))
            avg_clipped_expected_accept_length = float(sum(valid_clipped) / len(valid_clipped)) if valid_clipped else None
        else:
            avg_clipped_expected_accept_length = None

        # Expected accept length bucket accept length
        # bins: [0~1), [1~2), ..., [9~10)
        avg_accept_length_per_expected = {}
        bucket_sum = {i: 0.0 for i in range(10)}
        bucket_cnt = {i: 0 for i in range(10)}
        out_sum = 0.0
        out_cnt = 0

        if all_expected_accept_lengths_per_step and all_accept_lengths and len(all_expected_accept_lengths_per_step) == len(all_accept_lengths):
            for exp_val, acc_len in zip(all_expected_accept_lengths_per_step, all_accept_lengths):
                if exp_val is None:
                    out_sum += float(acc_len)
                    out_cnt += 1
                    continue
                try:
                    exp_f = float(exp_val)
                except Exception:
                    out_sum += float(acc_len)
                    out_cnt += 1
                    continue

                if 0.0 <= exp_f < 10.0:
                    idx = int(exp_f)  # 0~9
                    bucket_sum[idx] += float(acc_len)
                    bucket_cnt[idx] += 1
                else:
                    out_sum += float(acc_len)
                    out_cnt += 1

        # avg_width_per_depth
        for i in range(10):
            key = f"{i}-{i+1}"
            if bucket_cnt[i] > 0:
                avg_accept_length_per_expected[key] = {
                    "count": int(bucket_cnt[i]),
                    "avg_accept_length": float(bucket_sum[i] / bucket_cnt[i]),
                }
            else:
                avg_accept_length_per_expected[key] = {
                    "count": 0,
                    "avg_accept_length": None,
                }
        # (>=10, <0, None ) ( / )
        avg_accept_length_per_expected["out_of_range"] = {
            "count": int(out_cnt),
            "avg_accept_length": float(out_sum / out_cnt) if out_cnt > 0 else None,
        }

        def _finalize_post_reply_bucket(bucket: dict):
            out = {}
            for key, rec in sorted(bucket.items(), key=lambda kv: str(kv[0])):
                count = int(rec.get("count", 0) or 0)
                positive_count = int(rec.get("positive_count", 0) or 0)
                total_ms = float(rec.get("total_ms", 0.0) or 0.0)
                out[str(key)] = {
                    "count": count,
                    "positive_count": positive_count,
                    "total_ms": total_ms,
                    "avg_ms": (total_ms / count) if count > 0 else 0.0,
                    "avg_positive_ms": (total_ms / positive_count) if positive_count > 0 else 0.0,
                    "max_ms": float(rec.get("max_ms", 0.0) or 0.0),
                }
            return out

        _harvest_canceled_proactive_threads(force_join=True)
        if proactive_worker is not None:
            proactive_worker.shutdown()
        if recv_worker is not None:
            recv_worker.shutdown()
        if gpu_monitor_long_running and runner.gpu_monitor is not None:
            runner.gpu_monitor.stop_monitoring()

        # cost_per_token
        # : accept length
        print(
            "avg_accept_length: {:.4f}  acceptance_ratio_avg: {:.4f}".format(
                avg_accept_length,  # next token
                reported_acceptance_ratio_avg,
            )
        )
        # expected/ scaled expected ( avg_accept_length )
        try:
            _orig_exp = float(avg_expected_accept_length) if avg_expected_accept_length is not None else None
        except Exception:
            _orig_exp = None
        try:
            _scaled_exp = float(avg_scaled_expected_accept_length) if avg_scaled_expected_accept_length is not None else None
        except Exception:
            _scaled_exp = None
        print(
            "original_expected_accept_length: {}  scaled_expected_accept_length: {}".format(
                ("{:.4f}".format(_orig_exp) if _orig_exp is not None else "None"),
                ("{:.4f}".format(_scaled_exp) if _scaled_exp is not None else "None"),
            )
        )

        # Network latency and accept-length dictionary
        global_avg_draft = float(sum(global_draft_to_target_times) / len(global_draft_to_target_times)) if global_draft_to_target_times else None
        global_avg_target = float(sum(global_target_to_draft_times) / len(global_target_to_draft_times)) if global_target_to_draft_times else None
        global_avg_accept = float(sum(global_accept_lengths) / len(global_accept_lengths)) if global_accept_lengths else None
        global_accept_len_bucket_stats = {}
        for length, times in global_t2d_per_accept_len.items():
            if not times:
                continue
            global_accept_len_bucket_stats[int(length)] = {
                "count": len(times),
                "avg_target_to_draft_time_ms": float(sum(times) / len(times)),
                "min_target_to_draft_time_ms": float(min(times)),
                "max_target_to_draft_time_ms": float(max(times)),
            }
        # final_nnodes draft_to_target_time
        global_final_nnodes_bucket_stats = {}
        for nnodes, times in global_d2t_per_final_nnodes.items():
            if not times:
                continue
            global_final_nnodes_bucket_stats[int(nnodes)] = {
                "count": len(times),
                "avg_draft_to_target_time_ms": float(sum(times) / len(times)),
                "min_draft_to_target_time_ms": float(min(times)),
                "max_draft_to_target_time_ms": float(max(times)),
            }
        

        # Depth ( , )
        depth_statistics = {}
        depth_statistics_short = {}
        if runner.accumulated_depth_stats:
            for depth, stats in runner.accumulated_depth_stats.items():
                depth_statistics[depth] = {}
                depth_statistics_short[depth] = {}
                for key, values in stats.items():
                    if len(values) > 0:
                        mean_value = float(sum(values) / len(values))
                        depth_statistics[depth][key] = {
                            "count": len(values),
                            "min": float(min(values)),
                            "max": float(max(values)),
                            "mean": mean_value,
                            "sum": float(sum(values)),
                            "all_values": [float(v) for v in values]
                        }
                        # short mean
                        depth_statistics_short[depth][key] = mean_value
                    else:
                        depth_statistics[depth][key] = {
                            "count": 0,
                            "min": None,
                            "max": None,
                            "mean": None,
                            "sum": 0.0,
                            "all_values": []
                        }
                        depth_statistics_short[depth][key] = None

        # Width model.model() (print_timing_stats )
        width_timing_stats = {}
        if runner.accumulated_width_times:
            for width in sorted(runner.accumulated_width_times.keys()):
                times = runner.accumulated_width_times[width]
                count = len(times)
                total_time_ms = sum(times) * 1000
                avg_time_ms = (sum(times) / count) * 1000 if count > 0 else 0
                min_time_ms = min(times) * 1000
                max_time_ms = max(times) * 1000
                
                width_timing_stats[width] = {
                    "count": count,
                    "total_time_ms": float(total_time_ms),
                    "avg_time_ms": float(avg_time_ms),
                    "min_time_ms": float(min_time_ms),
                    "max_time_ms": float(max_time_ms),
                    "all_times_ms": [float(t * 1000) for t in times]
                }
        
        # Algorithm (objective_value, per_token_latency, per_token_cost )
        algorithm_stats = {}
        all_objective_values = []
        all_per_token_latency = []
        all_per_token_cost = []
        
        if runner.accumulated_depth_stats:
            for depth, stats in runner.accumulated_depth_stats.items():
                if "objective_value" in stats and len(stats["objective_value"]) > 0:
                    all_objective_values.extend(stats["objective_value"])
                if "per_token_latency" in stats and len(stats["per_token_latency"]) > 0:
                    all_per_token_latency.extend(stats["per_token_latency"])
                if "per_token_cost" in stats and len(stats["per_token_cost"]) > 0:
                    all_per_token_cost.extend(stats["per_token_cost"])
        
        if len(all_objective_values) > 0:
            algorithm_stats["avg_objective_value"] = float(sum(all_objective_values) / len(all_objective_values))
            algorithm_stats["min_objective_value"] = float(min(all_objective_values))
            algorithm_stats["max_objective_value"] = float(max(all_objective_values))
            algorithm_stats["count_objective_value"] = len(all_objective_values)
        else:
            algorithm_stats["avg_objective_value"] = None
            algorithm_stats["min_objective_value"] = None
            algorithm_stats["max_objective_value"] = None
            algorithm_stats["count_objective_value"] = 0
        
        if len(all_per_token_latency) > 0:
            algorithm_stats["avg_per_token_latency"] = float(sum(all_per_token_latency) / len(all_per_token_latency))
            algorithm_stats["min_per_token_latency"] = float(min(all_per_token_latency))
            algorithm_stats["max_per_token_latency"] = float(max(all_per_token_latency))
            algorithm_stats["count_per_token_latency"] = len(all_per_token_latency)
        else:
            algorithm_stats["avg_per_token_latency"] = None
            algorithm_stats["min_per_token_latency"] = None
            algorithm_stats["max_per_token_latency"] = None
            algorithm_stats["count_per_token_latency"] = 0
        
        if len(all_per_token_cost) > 0:
            algorithm_stats["avg_per_token_cost"] = float(sum(all_per_token_cost) / len(all_per_token_cost))
            algorithm_stats["min_per_token_cost"] = float(min(all_per_token_cost))
            algorithm_stats["max_per_token_cost"] = float(max(all_per_token_cost))
            algorithm_stats["count_per_token_cost"] = len(all_per_token_cost)
        else:
            algorithm_stats["avg_per_token_cost"] = None
            algorithm_stats["min_per_token_cost"] = None
            algorithm_stats["max_per_token_cost"] = None
            algorithm_stats["count_per_token_cost"] = 0
        
        # width
        if runner.accumulated_width_algorithm_times and len(runner.accumulated_width_algorithm_times) > 0:
            width_times = runner.accumulated_width_algorithm_times
            algorithm_stats["width_algorithm_time"] = {
                "total_seconds": float(sum(width_times)),
                "avg_seconds": float(sum(width_times) / len(width_times)),
                "min_seconds": float(min(width_times)),
                "max_seconds": float(max(width_times)),
                "count": len(width_times)
            }
        else:
            algorithm_stats["width_algorithm_time"] = {
                "total_seconds": 0.0,
                "avg_seconds": None,
                "min_seconds": None,
                "max_seconds": None,
                "count": 0
            }
        
        # final_nnodes
        if runner.accumulated_nnodes_algorithm_times and len(runner.accumulated_nnodes_algorithm_times) > 0:
            nnodes_times = runner.accumulated_nnodes_algorithm_times
            algorithm_stats["nnodes_algorithm_time"] = {
                "total_seconds": float(sum(nnodes_times)),
                "avg_seconds": float(sum(nnodes_times) / len(nnodes_times)),
                "min_seconds": float(min(nnodes_times)),
                "max_seconds": float(max(nnodes_times)),
                "count": len(nnodes_times)
            }
        else:
            algorithm_stats["nnodes_algorithm_time"] = {
                "total_seconds": 0.0,
                "avg_seconds": None,
                "min_seconds": None,
                "max_seconds": None,
                "count": 0
            }
        
        # total_algorithm_time (width_algorithm_time + nnodes_algorithm_time)
        width_total = algorithm_stats["width_algorithm_time"]["total_seconds"]
        nnodes_total = algorithm_stats["nnodes_algorithm_time"]["total_seconds"]
        total_algorithm_time = width_total + nnodes_total
        algorithm_stats["total_algorithm_time"] = float(total_algorithm_time)
        
        # (model_time )
        model_function_timing = {
            "actual_total_time_ms": float(runner.accumulated_model_total_time * 1000),
            "expected_total_time_ms": float(runner.accumulated_expected_model_total_time * 1000),
        }
        if runner.accumulated_model_total_time > 0:
            diff_ms = (runner.accumulated_expected_model_total_time - runner.accumulated_model_total_time) * 1000
            diff_percent = (diff_ms / (runner.accumulated_model_total_time * 1000)) * 100
            model_function_timing["diff_ms"] = float(diff_ms)
            model_function_timing["diff_percent"] = float(diff_percent)
        else:
            model_function_timing["diff_ms"] = None
            model_function_timing["diff_percent"] = None

        # Latency (raw )
        latency_statistics = {}
        if latency_data:
            # latency
            tree_build_times = [d["tree_build_time_ms"] for d in latency_data if d["tree_build_time_ms"] is not None]
            tree_build_wall_times = [
                d["tree_build_wall_time_ms"]
                for d in latency_data
                if d.get("tree_build_wall_time_ms") is not None
            ]
            draft_to_target_times = [d["draft_to_target_time_ms"] for d in latency_data if d["draft_to_target_time_ms"] is not None]
            target_verification_times = [d["target_verification_time_ms"] for d in latency_data if d["target_verification_time_ms"] is not None]
            target_to_draft_times = [d["target_to_draft_time_ms"] for d in latency_data if d["target_to_draft_time_ms"] is not None]
            payload_build_times = [d["payload_build_ms"] for d in latency_data if d.get("payload_build_ms") is not None]
            send_json_times = [d["send_json_ms"] for d in latency_data if d.get("send_json_ms") is not None]
            recv_submit_times = [d["recv_submit_ms"] for d in latency_data if d.get("recv_submit_ms") is not None]
            recv_wait_times = [d["recv_wait_ms"] for d in latency_data if d.get("recv_wait_ms") is not None]
            proactive_path_select_times = [d["proactive_path_select_ms"] for d in latency_data if d.get("proactive_path_select_ms") is not None]
            proactive_decision_times = [d["proactive_decision_ms"] for d in latency_data if d.get("proactive_decision_ms") is not None]
            monitor_stop_times = [d["monitor_stop_ms"] for d in latency_data if d.get("monitor_stop_ms") is not None]
            stats_update_times = [d["stats_update_ms"] for d in latency_data if d.get("stats_update_ms") is not None]
            input_update_times = [d["input_update_ms"] for d in latency_data if d.get("input_update_ms") is not None]
            tree_model_forward_times = [d["tree_model_forward_ms"] for d in latency_data if d.get("tree_model_forward_ms") is not None]
            tree_width_algo_times = [d["tree_width_algo_ms"] for d in latency_data if d.get("tree_width_algo_ms") is not None]
            tree_nnodes_algo_times = [d["tree_nnodes_algo_ms"] for d in latency_data if d.get("tree_nnodes_algo_ms") is not None]
            tree_mask_build_times = [d["tree_mask_build_ms"] for d in latency_data if d.get("tree_mask_build_ms") is not None]
            tree_finalize_times = [d["tree_finalize_ms"] for d in latency_data if d.get("tree_finalize_ms") is not None]
            tree_budget_wait_times = [d["tree_budget_wait_ms"] for d in latency_data if d.get("tree_budget_wait_ms") is not None]
            proactive_budget_times = [d["proactive_budget_ms"] for d in latency_data if d.get("proactive_budget_ms") is not None]
            proactive_wall_elapsed_times = [
                d["proactive_wall_elapsed_ms"]
                for d in latency_data
                if d.get("proactive_wall_elapsed_ms") is not None
            ]
            proactive_compute_elapsed_times = [
                d["proactive_compute_elapsed_ms"]
                for d in latency_data
                if d.get("proactive_compute_elapsed_ms") is not None
            ]
            proactive_compute_hidden_before_reply_times = [
                d["proactive_compute_hidden_before_reply_ms"]
                for d in latency_data
                if d.get("proactive_compute_hidden_before_reply_ms") is not None
            ]
            proactive_compute_after_reply_times = [
                d["proactive_compute_after_reply_ms"]
                for d in latency_data
                if d.get("proactive_compute_after_reply_ms") is not None
            ]
            proactive_budget_wait_times = [
                d["proactive_budget_wait_ms"]
                for d in latency_data
                if d.get("proactive_budget_wait_ms") is not None
            ]
            proactive_resume_after_reply_times = [
                d["proactive_resume_after_reply_ms"]
                for d in latency_data
                if d.get("proactive_resume_after_reply_ms") is not None
            ]
            proactive_post_reply_wait_times = [
                d["proactive_post_reply_wait_ms"]
                for d in latency_data
                if d.get("proactive_post_reply_wait_ms") is not None
            ]
            proactive_cancel_immediate_join_times = [
                d["proactive_cancel_immediate_join_ms"]
                for d in latency_data
                if d.get("proactive_cancel_immediate_join_ms") is not None
            ]
            proactive_cancel_join_before_tree_build_times = [
                d["proactive_cancel_join_before_tree_build_ms"]
                for d in latency_data
                if d.get("proactive_cancel_join_before_tree_build_ms") is not None
            ]
            proactive_cancel_tree_build_overlap_times = [
                d["proactive_cancel_tree_build_overlap_ms"]
                for d in latency_data
                if d.get("proactive_cancel_tree_build_overlap_ms") is not None
            ]
            proactive_expected_gain_times = [
                d["proactive_expected_gain_ms"]
                for d in latency_data
                if d.get("proactive_expected_gain_ms") is not None
            ]
            proactive_expected_loss_times = [
                d["proactive_expected_loss_ms"]
                for d in latency_data
                if d.get("proactive_expected_loss_ms") is not None
            ]
            proactive_post_reply_depth_values = [
                d["proactive_post_reply_tree_depth"]
                for d in latency_data
                if d.get("proactive_post_reply_tree_depth") is not None
            ]
            proactive_post_reply_avg_width_values = [
                d["proactive_post_reply_avg_width"]
                for d in latency_data
                if d.get("proactive_post_reply_avg_width") is not None
            ]
            
            def calc_stats(values):
                if not values:
                    return {"min": None, "max": None, "mean": None, "count": 0}
                return {
                    "min": float(min(values)),
                    "max": float(max(values)),
                    "mean": float(sum(values) / len(values)),
                    "count": len(values)
                }
            
            latency_statistics = {
                "tree_build": {
                    "raw": tree_build_times,
                    "stats": calc_stats(tree_build_times)
                },
                "tree_build_wall": {
                    "raw": tree_build_wall_times,
                    "stats": calc_stats(tree_build_wall_times)
                },
                "draft_to_target": {
                    "raw": draft_to_target_times,
                    "stats": calc_stats(draft_to_target_times)
                },
                "target_verification": {
                    "raw": target_verification_times,
                    "stats": calc_stats(target_verification_times)
                },
                "target_to_draft": {
                    "raw": target_to_draft_times,
                    "stats": calc_stats(target_to_draft_times)
                },
                "payload_build": {
                    "raw": payload_build_times,
                    "stats": calc_stats(payload_build_times)
                },
                "send_json": {
                    "raw": send_json_times,
                    "stats": calc_stats(send_json_times)
                },
                "recv_submit": {
                    "raw": recv_submit_times,
                    "stats": calc_stats(recv_submit_times)
                },
                "recv_wait": {
                    "raw": recv_wait_times,
                    "stats": calc_stats(recv_wait_times)
                },
                "proactive_path_select": {
                    "raw": proactive_path_select_times,
                    "stats": calc_stats(proactive_path_select_times)
                },
                "proactive_decision": {
                    "raw": proactive_decision_times,
                    "stats": calc_stats(proactive_decision_times)
                },
                "monitor_stop": {
                    "raw": monitor_stop_times,
                    "stats": calc_stats(monitor_stop_times)
                },
                "stats_update": {
                    "raw": stats_update_times,
                    "stats": calc_stats(stats_update_times)
                },
                "input_update": {
                    "raw": input_update_times,
                    "stats": calc_stats(input_update_times)
                },
                "tree_model_forward": {
                    "raw": tree_model_forward_times,
                    "stats": calc_stats(tree_model_forward_times)
                },
                "tree_width_algo": {
                    "raw": tree_width_algo_times,
                    "stats": calc_stats(tree_width_algo_times)
                },
                "tree_nnodes_algo": {
                    "raw": tree_nnodes_algo_times,
                    "stats": calc_stats(tree_nnodes_algo_times)
                },
                "tree_mask_build": {
                    "raw": tree_mask_build_times,
                    "stats": calc_stats(tree_mask_build_times)
                },
                "tree_finalize": {
                    "raw": tree_finalize_times,
                    "stats": calc_stats(tree_finalize_times)
                },
                "tree_budget_wait": {
                    "raw": tree_budget_wait_times,
                    "stats": calc_stats(tree_budget_wait_times)
                },
                "proactive_budget": {
                    "raw": proactive_budget_times,
                    "stats": calc_stats(proactive_budget_times)
                },
                "proactive_wall_elapsed": {
                    "raw": proactive_wall_elapsed_times,
                    "stats": calc_stats(proactive_wall_elapsed_times)
                },
                "proactive_compute_elapsed": {
                    "raw": proactive_compute_elapsed_times,
                    "stats": calc_stats(proactive_compute_elapsed_times)
                },
                "proactive_compute_hidden_before_reply": {
                    "raw": proactive_compute_hidden_before_reply_times,
                    "stats": calc_stats(proactive_compute_hidden_before_reply_times)
                },
                "proactive_compute_after_reply": {
                    "raw": proactive_compute_after_reply_times,
                    "stats": calc_stats(proactive_compute_after_reply_times)
                },
                "proactive_budget_wait": {
                    "raw": proactive_budget_wait_times,
                    "stats": calc_stats(proactive_budget_wait_times)
                },
                "proactive_resume_after_reply": {
                    "raw": proactive_resume_after_reply_times,
                    "stats": calc_stats(proactive_resume_after_reply_times)
                },
                "proactive_post_reply_wait": {
                    "raw": proactive_post_reply_wait_times,
                    "stats": calc_stats(proactive_post_reply_wait_times)
                },
                "proactive_cancel_immediate_join": {
                    "raw": proactive_cancel_immediate_join_times,
                    "stats": calc_stats(proactive_cancel_immediate_join_times)
                },
                "proactive_cancel_join_before_tree_build": {
                    "raw": proactive_cancel_join_before_tree_build_times,
                    "stats": calc_stats(proactive_cancel_join_before_tree_build_times)
                },
                "proactive_cancel_tree_build_overlap": {
                    "raw": proactive_cancel_tree_build_overlap_times,
                    "stats": calc_stats(proactive_cancel_tree_build_overlap_times)
                },
                "proactive_expected_gain": {
                    "raw": proactive_expected_gain_times,
                    "stats": calc_stats(proactive_expected_gain_times)
                },
                "proactive_expected_loss": {
                    "raw": proactive_expected_loss_times,
                    "stats": calc_stats(proactive_expected_loss_times)
                },
                "proactive_post_reply_tree_depth": {
                    "raw": proactive_post_reply_depth_values,
                    "stats": calc_stats(proactive_post_reply_depth_values)
                },
                "proactive_post_reply_avg_width": {
                    "raw": proactive_post_reply_avg_width_values,
                    "stats": calc_stats(proactive_post_reply_avg_width_values)
                }
            }
        
        # Width draft latency ( vs )
        per_width_draft_latency = {}
        if runner.accumulated_width_times and runner.profile_data:
            for width in sorted(runner.accumulated_width_times.keys()):
                actual_times = runner.accumulated_width_times[width]
                actual_times_ms = [t * 1000.0 for t in actual_times]
                
                if actual_times_ms:
                    actual_avg_ms = float(sum(actual_times_ms) / len(actual_times_ms))
                    actual_min_ms = float(min(actual_times_ms))
                    actual_max_ms = float(max(actual_times_ms))
                else:
                    actual_avg_ms = None
                    actual_min_ms = None
                    actual_max_ms = None
                
                width_str = str(width)
                if width_str in runner.profile_data:
                    predicted_time_ms = runner.profile_data[width_str].get("model_call_avg_time_ms", None)
                else:
                    predicted_time_ms = None
                
                per_width_draft_latency[width_str] = {
                    "predicted_time_ms": predicted_time_ms,
                    "actual_avg_ms": actual_avg_ms,
                    "actual_min_ms": actual_min_ms,
                    "actual_max_ms": actual_max_ms,
                    "count": len(actual_times)
                }
        
        # per_width_draft_latency latency_statistics
        if per_width_draft_latency:
            latency_statistics["per_width_draft_latency"] = per_width_draft_latency
        
        # (KByte )
        data_transfer_stats = {}
        if latency_data:
            draft_to_target_bytes_list = [
                d.get("draft_to_target_bytes")
                for d in latency_data
                if d.get("draft_to_target_bytes") is not None
            ]
            target_to_draft_bytes_list = [
                d.get("target_to_draft_bytes")
                for d in latency_data
                if d.get("target_to_draft_bytes") is not None
            ]
            
            # Byte KByte
            draft_to_target_kbytes = [b / 1024.0 for b in draft_to_target_bytes_list]
            target_to_draft_kbytes = [b / 1024.0 for b in target_to_draft_bytes_list]
            
            def calc_transfer_stats(kbytes_list, bytes_list):
                if not kbytes_list:
                    return {
                        "total_kbytes": 0.0,
                        "avg_kbytes": None,
                        "min_kbytes": None,
                        "max_kbytes": None,
                        "count": 0,
                        "raw_kbytes": []
                    }
                return {
                    "total_kbytes": float(sum(kbytes_list)),
                    "avg_kbytes": float(sum(kbytes_list) / len(kbytes_list)),
                    "min_kbytes": float(min(kbytes_list)),
                    "max_kbytes": float(max(kbytes_list)),
                    "count": len(kbytes_list),
                    "raw_kbytes": [float(kb) for kb in kbytes_list]
                }
            
            data_transfer_stats = {
                "draft_to_target": calc_transfer_stats(draft_to_target_kbytes, draft_to_target_bytes_list),
                "target_to_draft": calc_transfer_stats(target_to_draft_kbytes, target_to_draft_bytes_list),
                "total_kbytes": float(sum(draft_to_target_kbytes) + sum(target_to_draft_kbytes)) if draft_to_target_kbytes and target_to_draft_kbytes else 0.0
            }

        draft_model_name = draft_model_path.split("/")[-1] if "/" in draft_model_path else draft_model_path
        base_model_name = base_model_path.split("/")[-1] if "/" in base_model_path else base_model_path
        
        script_dir = os.path.dirname(__file__)
        parent_dir = os.path.dirname(script_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # answer_file ,
        if answer_file:
            output_dir = os.path.dirname(os.path.expanduser(answer_file))
            # .json +
            base_filename = os.path.splitext(os.path.basename(answer_file))[0]
            output_file = os.path.join(output_dir, f"{base_filename}_results_{timestamp}.json")
        else:
            output_dir = os.path.join(parent_dir, "result", "experiments_draft")
            if device_name:
                output_file = os.path.join(output_dir, f"draft_{draft_model_name}_{device_name}_{bench_name}_nodes{nodes}_{timestamp}_results.json")
            else:
                output_file = os.path.join(output_dir, f"draft_{draft_model_name}_{bench_name}_nodes{nodes}_{timestamp}_results.json")
        
        os.makedirs(output_dir, exist_ok=True)
        
        network_info = {}
        try:
            if PSUTIL_AVAILABLE:
                # psutil
                interfaces = psutil.net_if_addrs()
                stats = psutil.net_if_stats()
                active_interfaces = []
                for interface_name, addrs in interfaces.items():
                    if interface_name in stats and stats[interface_name].isup:
                        interface_type = "unknown"
                        if "wlan" in interface_name.lower() or "wifi" in interface_name.lower() or "wireless" in interface_name.lower():
                            interface_type = "wifi"
                        elif "eth" in interface_name.lower() or "enp" in interface_name.lower() or "eno" in interface_name.lower():
                            interface_type = "ethernet"
                        elif "lo" in interface_name.lower():
                            interface_type = "loopback"
                        
                        active_interfaces.append({
                            "name": interface_name,
                            "type": interface_type,
                            "speed_mbps": stats[interface_name].speed if stats[interface_name].speed > 0 else None
                        })
                
                if active_interfaces:
                    primary_interface = max(active_interfaces, key=lambda x: x["speed_mbps"] if x["speed_mbps"] else 0) if active_interfaces else active_interfaces[0]
                    network_info = {
                        "primary_interface": primary_interface["name"],
                        "interface_type": primary_interface["type"],
                        "speed_mbps": primary_interface["speed_mbps"],
                        "all_active_interfaces": active_interfaces
                    }
        except Exception as e:
            network_info = {"error": str(e)}
        
        # GPU
        gpu_info = {}
        try:
            if torch.cuda.is_available():
                # PyTorch GPU
                gpu_info["cuda_available"] = True
                gpu_info["gpu_count"] = torch.cuda.device_count()
                gpu_info["current_device"] = torch.cuda.current_device()
                gpu_info["device_name"] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
                
                # nvidia-smi
                try:
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader,nounits'],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        if lines and lines[0].strip():
                            parts = lines[0].split(', ')
                            if len(parts) >= 3:
                                gpu_info["gpu_name"] = parts[0].strip()
                                gpu_info["total_memory_mb"] = int(parts[1].strip()) if parts[1].strip().isdigit() else None
                                gpu_info["driver_version"] = parts[2].strip()
                except Exception:
                    pass
            else:
                gpu_info["cuda_available"] = False
        except Exception as e:
            gpu_info["error"] = str(e)
        
        # GPU (draft_gpu_data draft_timing_data )
        draft_gpu_summary = {}
        if draft_gpu_data and draft_timing_data:
            # GPU
            all_gpu_stats = {}
            for step_data in draft_gpu_data:
                for gpu_id, stats in step_data["gpu_stats"].items():
                    if gpu_id not in all_gpu_stats:
                        all_gpu_stats[gpu_id] = {
                            'memory_used_mb': [],
                            'utilization_percent': [],
                            'temperature_c': [],
                            'memory_used_percent': [],
                            'power_draw_w': [],
                            'power_limit_w': [],
                            'power_usage_percent': []
                        }
                    
                    all_gpu_stats[gpu_id]['memory_used_mb'].extend([stats['memory_used_mb']['avg']])
                    all_gpu_stats[gpu_id]['utilization_percent'].extend([stats['utilization_percent']['avg']])
                    all_gpu_stats[gpu_id]['temperature_c'].extend([stats['temperature_c']['avg']])
                    all_gpu_stats[gpu_id]['memory_used_percent'].extend([stats['memory_used_percent']['avg']])
                    
                    if 'power_draw_w' in stats:
                        all_gpu_stats[gpu_id]['power_draw_w'].extend([stats['power_draw_w']['avg']])
                    if 'power_limit_w' in stats:
                        all_gpu_stats[gpu_id]['power_limit_w'].extend([stats['power_limit_w']])
                    if 'power_usage_percent' in stats:
                        all_gpu_stats[gpu_id]['power_usage_percent'].extend([stats['power_usage_percent']['avg']])
            
            for gpu_id, data in all_gpu_stats.items():
                draft_gpu_summary[gpu_id] = {
                    'memory_used_mb': {
                        'min': min(data['memory_used_mb']),
                        'max': max(data['memory_used_mb']),
                        'avg': sum(data['memory_used_mb']) / len(data['memory_used_mb'])
                    },
                    'utilization_percent': {
                        'min': min(data['utilization_percent']),
                        'max': max(data['utilization_percent']),
                        'avg': sum(data['utilization_percent']) / len(data['utilization_percent'])
                    },
                    'temperature_c': {
                        'min': min(data['temperature_c']),
                        'max': max(data['temperature_c']),
                        'avg': sum(data['temperature_c']) / len(data['temperature_c'])
                    },
                    'memory_used_percent': {
                        'min': min(data['memory_used_percent']),
                        'max': max(data['memory_used_percent']),
                        'avg': sum(data['memory_used_percent']) / len(data['memory_used_percent'])
                    }
                }
                
                if data['power_draw_w']:
                    draft_gpu_summary[gpu_id]['power_draw_w'] = {
                        'min': min(data['power_draw_w']),
                        'max': max(data['power_draw_w']),
                        'avg': sum(data['power_draw_w']) / len(data['power_draw_w'])
                    }
                if data['power_limit_w']:
                    # power_limit_w
                    draft_gpu_summary[gpu_id]['power_limit_w'] = data['power_limit_w'][0]
                if data['power_usage_percent']:
                    draft_gpu_summary[gpu_id]['power_usage_percent'] = {
                        'min': min(data['power_usage_percent']),
                        'max': max(data['power_usage_percent']),
                        'avg': sum(data['power_usage_percent']) / len(data['power_usage_percent'])
                    }
                
                # GPU 
                if data['power_draw_w'] and draft_timing_data:
                    total_energy_joules = 0.0
                    step_energies = []
                    
                    # draft_gpu_data draft_timing_data step
                    for step_idx, step_data in enumerate(draft_gpu_data):
                        if step_idx < len(draft_timing_data):
                            # step GPU power_draw_w
                            if gpu_id in step_data.get('gpu_stats', {}):
                                step_power_avg = step_data['gpu_stats'][gpu_id].get('power_draw_w', {}).get('avg', 0.0)
                                step_time = draft_timing_data[step_idx].get('total_time_seconds', 0.0)
                                step_energy = step_power_avg * step_time  # = (Joule)
                                step_energies.append(step_energy)
                                total_energy_joules += step_energy
                    
                    # start_monitoring() (draft call )
                    draft_call_count = grand_total_steps  # : grand_total_steps
                    if draft_gpu_data:
                        last_step_data = draft_gpu_data[-1]
                        if last_step_data.get('monitor_call_count') is not None:
                            draft_call_count = last_step_data.get('monitor_call_count')
                    
                    if step_energies and draft_call_count > 0:
                        draft_gpu_summary[gpu_id]['energy'] = {
                            'total_energy_joules': float(total_energy_joules),
                            'total_energy_kwh': float(total_energy_joules / 3600000.0),  # kWh (1 kWh = 3,600,000 J)
                            'avg_energy_per_call_joules': float(total_energy_joules / draft_call_count),
                            'avg_energy_per_call_kwh': float((total_energy_joules / draft_call_count) / 3600000.0),
                            'min_energy_per_call_joules': float(min(step_energies)) if step_energies else 0.0,
                            'max_energy_per_call_joules': float(max(step_energies)) if step_energies else 0.0,
                            'num_calls': int(draft_call_count)  # start_monitoring()
                        }

        proactive_draft_gpu_summary = {}
        if proactive_gpu_data:
            proactive_all_gpu_stats = {}
            for step_data in proactive_gpu_data:
                for gpu_id, stats in step_data.get("gpu_stats", {}).items():
                    if gpu_id not in proactive_all_gpu_stats:
                        proactive_all_gpu_stats[gpu_id] = {
                            'memory_used_mb': [],
                            'utilization_percent': [],
                            'temperature_c': [],
                            'memory_used_percent': [],
                            'power_draw_w': [],
                            'power_limit_w': [],
                            'power_usage_percent': [],
                            'energy_joules': [],
                            'used_energy_joules': [],
                            'unused_energy_joules': [],
                        }
                    proactive_all_gpu_stats[gpu_id]['memory_used_mb'].append(stats['memory_used_mb']['avg'])
                    proactive_all_gpu_stats[gpu_id]['utilization_percent'].append(stats['utilization_percent']['avg'])
                    proactive_all_gpu_stats[gpu_id]['temperature_c'].append(stats['temperature_c']['avg'])
                    proactive_all_gpu_stats[gpu_id]['memory_used_percent'].append(stats['memory_used_percent']['avg'])
                    if 'power_draw_w' in stats:
                        proactive_all_gpu_stats[gpu_id]['power_draw_w'].append(stats['power_draw_w']['avg'])
                    if 'power_limit_w' in stats:
                        proactive_all_gpu_stats[gpu_id]['power_limit_w'].append(stats['power_limit_w'])
                    if 'power_usage_percent' in stats:
                        proactive_all_gpu_stats[gpu_id]['power_usage_percent'].append(stats['power_usage_percent']['avg'])
                    power_info = stats.get("power_draw_w", {}) if isinstance(stats, dict) else {}
                    try:
                        power_avg_w = float(power_info.get("avg", 0.0) if isinstance(power_info, dict) else 0.0)
                    except Exception:
                        power_avg_w = 0.0
                    energy_j = max(0.0, power_avg_w) * max(
                        0.0,
                        float(step_data.get("total_time_seconds", 0.0) or 0.0),
                    )
                    proactive_all_gpu_stats[gpu_id]['energy_joules'].append(energy_j)
                    if step_data.get("used"):
                        proactive_all_gpu_stats[gpu_id]['used_energy_joules'].append(energy_j)
                    else:
                        proactive_all_gpu_stats[gpu_id]['unused_energy_joules'].append(energy_j)

            for gpu_id, data in proactive_all_gpu_stats.items():
                total_energy_joules = float(sum(data['energy_joules']))
                used_energy_joules = float(sum(data['used_energy_joules']))
                unused_energy_joules = float(sum(data['unused_energy_joules']))
                proactive_draft_gpu_summary[gpu_id] = {
                    'memory_used_mb': {
                        'min': min(data['memory_used_mb']),
                        'max': max(data['memory_used_mb']),
                        'avg': sum(data['memory_used_mb']) / len(data['memory_used_mb']),
                    },
                    'utilization_percent': {
                        'min': min(data['utilization_percent']),
                        'max': max(data['utilization_percent']),
                        'avg': sum(data['utilization_percent']) / len(data['utilization_percent']),
                    },
                    'temperature_c': {
                        'min': min(data['temperature_c']),
                        'max': max(data['temperature_c']),
                        'avg': sum(data['temperature_c']) / len(data['temperature_c']),
                    },
                    'memory_used_percent': {
                        'min': min(data['memory_used_percent']),
                        'max': max(data['memory_used_percent']),
                        'avg': sum(data['memory_used_percent']) / len(data['memory_used_percent']),
                    },
                    'energy': {
                        'total_energy_joules': total_energy_joules,
                        'total_energy_kwh': total_energy_joules / 3600000.0,
                        'used_energy_joules': used_energy_joules,
                        'used_energy_kwh': used_energy_joules / 3600000.0,
                        'unused_energy_joules': unused_energy_joules,
                        'unused_energy_kwh': unused_energy_joules / 3600000.0,
                        'num_calls': int(len(data['energy_joules'])),
                    },
                }
                if data['power_draw_w']:
                    proactive_draft_gpu_summary[gpu_id]['power_draw_w'] = {
                        'min': min(data['power_draw_w']),
                        'max': max(data['power_draw_w']),
                        'avg': sum(data['power_draw_w']) / len(data['power_draw_w']),
                    }
                if data['power_limit_w']:
                    proactive_draft_gpu_summary[gpu_id]['power_limit_w'] = data['power_limit_w'][0]
                if data['power_usage_percent']:
                    proactive_draft_gpu_summary[gpu_id]['power_usage_percent'] = {
                        'min': min(data['power_usage_percent']),
                        'max': max(data['power_usage_percent']),
                        'avg': sum(data['power_usage_percent']) / len(data['power_usage_percent']),
                    }
        
        # CPU
        draft_cpu_power_summary = {}
        if draft_cpu_power_data and draft_timing_data:
            # CPU
            all_cpu_power_values = []
            for step_idx, step_data in enumerate(draft_cpu_power_data):
                if "cpu_power_stats" in step_data and step_data["cpu_power_stats"]:
                    cpu_stats = step_data["cpu_power_stats"]
                    if "cpu_power_w" in cpu_stats and "avg" in cpu_stats["cpu_power_w"]:
                        all_cpu_power_values.append(cpu_stats["cpu_power_w"]["avg"])

        
            
            if len(all_cpu_power_values) > 0:
                draft_cpu_power_summary = {
                    'cpu_power_w': {
                        'min': min(all_cpu_power_values),
                        'max': max(all_cpu_power_values),
                        'avg': sum(all_cpu_power_values) / len(all_cpu_power_values),
                        'count': len(all_cpu_power_values)
                    }
                }
                
                # CPU 
                total_cpu_energy_joules = 0.0
                step_cpu_energies = []
                
                # draft_cpu_power_data draft_timing_data step
                for step_idx, step_data in enumerate(draft_cpu_power_data):
                    if step_idx < len(draft_timing_data):
                        if "cpu_power_stats" in step_data and step_data["cpu_power_stats"]:
                            cpu_stats = step_data["cpu_power_stats"]
                            if "cpu_power_w" in cpu_stats and "avg" in cpu_stats["cpu_power_w"]:
                                step_power_avg = cpu_stats["cpu_power_w"]["avg"]
                                step_time = draft_timing_data[step_idx].get('total_time_seconds', 0.0)
                                step_energy = step_power_avg * step_time  # = (Joule)
                                step_cpu_energies.append(step_energy)
                                total_cpu_energy_joules += step_energy
                
                # start_monitoring() (draft call )
                cpu_draft_call_count = grand_total_steps  # : grand_total_steps
                if draft_cpu_power_data:
                    last_step_data = draft_cpu_power_data[-1]
                    if last_step_data.get('monitor_call_count') is not None:
                        cpu_draft_call_count = last_step_data.get('monitor_call_count')
                
                if step_cpu_energies and cpu_draft_call_count > 0:
                    draft_cpu_power_summary['energy'] = {
                        'total_energy_joules': float(total_cpu_energy_joules),
                        'total_energy_kwh': float(total_cpu_energy_joules / 3600000.0),  # kWh (1 kWh = 3,600,000 J)
                        'avg_energy_per_call_joules': float(total_cpu_energy_joules / cpu_draft_call_count),
                        'avg_energy_per_call_kwh': float((total_cpu_energy_joules / cpu_draft_call_count) / 3600000.0),
                        'min_energy_per_call_joules': float(min(step_cpu_energies)) if step_cpu_energies else 0.0,
                        'max_energy_per_call_joules': float(max(step_cpu_energies)) if step_cpu_energies else 0.0,
                        'num_calls': int(cpu_draft_call_count)  # start_monitoring()
                    }
        
        
        # Draft (GPU/CPU)
        foreground_draft_gpu_energy_kwh = 0.0
        for gpu_entry in draft_gpu_summary.values():
            energy_info = gpu_entry.get("energy", {}) if isinstance(gpu_entry, dict) else {}
            foreground_draft_gpu_energy_kwh += float(energy_info.get("total_energy_kwh", 0.0) or 0.0)
        proactive_draft_gpu_energy_kwh = 0.0
        proactive_draft_gpu_used_energy_kwh = 0.0
        proactive_draft_gpu_unused_energy_kwh = 0.0
        for gpu_entry in proactive_draft_gpu_summary.values():
            energy_info = gpu_entry.get("energy", {}) if isinstance(gpu_entry, dict) else {}
            proactive_draft_gpu_energy_kwh += float(energy_info.get("total_energy_kwh", 0.0) or 0.0)
            proactive_draft_gpu_used_energy_kwh += float(energy_info.get("used_energy_kwh", 0.0) or 0.0)
            proactive_draft_gpu_unused_energy_kwh += float(energy_info.get("unused_energy_kwh", 0.0) or 0.0)
        total_draft_gpu_energy_kwh = float(foreground_draft_gpu_energy_kwh + proactive_draft_gpu_energy_kwh)
        gpu_monitor_enabled = bool(runner.gpu_monitor is not None)
        gpu_monitor_sample_count = len(draft_gpu_data) if isinstance(draft_gpu_data, list) else 0
        gpu_monitor_power_draw_available = any(
            isinstance(gpu_entry, dict)
            and isinstance(gpu_entry.get("power_draw_w", None), dict)
            and (gpu_entry["power_draw_w"].get("avg", None) is not None)
            for gpu_entry in (draft_gpu_summary.values() if isinstance(draft_gpu_summary, dict) else [])
        )
        if not gpu_monitor_enabled:
            gpu_monitor_status = "disabled"
        elif gpu_monitor_sample_count <= 0:
            gpu_monitor_status = "no_samples"
        elif gpu_monitor_power_draw_available:
            gpu_monitor_status = "ok"
        else:
            gpu_monitor_status = "no_power_draw"
        total_draft_cpu_energy_kwh = float(
            (draft_cpu_power_summary.get("energy", {}) or {}).get("total_energy_kwh", 0.0) or 0.0
        )
        total_draft_energy_kwh = float(total_draft_gpu_energy_kwh + total_draft_cpu_energy_kwh)

        # total_cost cost_per_token
        total_new_tokens = grand_total_accepted + grand_total_steps
        draft_per_sec_cost = draft_per_hour_cost / 3600.0 if draft_per_hour_cost else 0.0
        target_per_sec_cost = target_per_hour_cost / 3600.0 if target_per_hour_cost else 0.0
        draft_electricity_cost_per_kwh = float(getattr(runner, "draft_electricity_cost_per_kwh", 0.09))
        
        # tree_build 
        total_tree_build_time_sec = 0.0
        if latency_statistics and "tree_build" in latency_statistics and "stats" in latency_statistics["tree_build"]:
            tree_build_stats = latency_statistics["tree_build"]["stats"]
            if tree_build_stats.get("mean") is not None and tree_build_stats.get("count", 0) > 0:
                total_tree_build_time_sec = (tree_build_stats["mean"] * tree_build_stats["count"]) / 1000.0
        total_tree_build_wall_time_sec = 0.0
        if latency_statistics and "tree_build_wall" in latency_statistics and "stats" in latency_statistics["tree_build_wall"]:
            tree_build_wall_stats = latency_statistics["tree_build_wall"]["stats"]
            if tree_build_wall_stats.get("mean") is not None and tree_build_wall_stats.get("count", 0) > 0:
                total_tree_build_wall_time_sec = (
                    tree_build_wall_stats["mean"] * tree_build_wall_stats["count"]
                ) / 1000.0
        
        # target_verification 
        total_target_verification_time_sec = 0.0
        if latency_statistics and "target_verification" in latency_statistics and "stats" in latency_statistics["target_verification"]:
            target_verification_stats = latency_statistics["target_verification"]["stats"]
            if target_verification_stats.get("mean") is not None and target_verification_stats.get("count", 0) > 0:
                total_target_verification_time_sec = (target_verification_stats["mean"] * target_verification_stats["count"]) / 1000.0
        
        # total_cost (proactive tree_build_time )
        total_draft_time_sec = total_tree_build_time_sec
        if no_draft_cost:
            draft_cost = 0.0
            proactive_draft_cost = 0.0
        elif str(objective_metric).lower() in {"total_cost", "api_cost"}:
            if server_only_mode:
                # server-only draft/target server ,
                # target_per_sec_cost .
                proactive_draft_cost = max(0.0, float(proactive_total_time_sec)) * max(0.0, float(target_per_sec_cost))
                draft_cost = max(0.0, float(total_draft_time_sec)) * max(0.0, float(target_per_sec_cost))
            elif bool(getattr(runner, "bill_draft_as_target_gpu", False)):
                proactive_draft_cost = max(0.0, float(proactive_total_time_sec)) * max(0.0, float(target_per_sec_cost))
                draft_cost = max(0.0, float(total_draft_time_sec)) * max(0.0, float(target_per_sec_cost))
            else:
                # total_cost draft GPU (kWh) * ($/kWh)
                draft_cost = max(0.0, float(total_draft_gpu_energy_kwh)) * max(0.0, draft_electricity_cost_per_kwh)
                proactive_draft_cost = (
                    max(0.0, float(proactive_draft_gpu_energy_kwh))
                    * max(0.0, draft_electricity_cost_per_kwh)
                )
        else:
            proactive_draft_cost = max(0.0, float(proactive_total_time_sec)) * max(0.0, float(draft_per_sec_cost))
            draft_cost = max(0.0, float(total_draft_time_sec)) * max(0.0, float(draft_per_sec_cost))
        server_only_transfer_time_sec = (
            (sum(global_draft_to_target_times) + sum(global_target_to_draft_times)) / 1000.0
            if server_only_mode
            else 0.0
        )
        target_billed_time_sec = total_target_verification_time_sec
        target_cost = target_billed_time_sec * target_per_sec_cost
        total_d2t_bytes = max(0.0, float(sum(run_draft_to_target_bytes))) if run_draft_to_target_bytes else 0.0
        total_t2d_bytes = max(0.0, float(sum(run_target_to_draft_bytes))) if run_target_to_draft_bytes else 0.0
        total_transfer_bytes = total_d2t_bytes + total_t2d_bytes
        bytes_per_gb = float(1024 ** 3)
        user_communication_inbound_cost = max(0.0, (
            (total_d2t_bytes / bytes_per_gb) * float(user_communication_cost_per_gb)
            if bytes_per_gb > 0
            else 0.0
        ))
        user_communication_outbound_cost = max(0.0, (
            (total_t2d_bytes / bytes_per_gb) * float(user_communication_cost_per_gb)
            if bytes_per_gb > 0
            else 0.0
        ))
        cloud_outbound_cost = max(0.0, (
            (total_t2d_bytes / bytes_per_gb) * float(cloud_outbound_cost_per_gb)
            if bytes_per_gb > 0
            else 0.0
        ))
        total_outbound_comm_cost = user_communication_outbound_cost + cloud_outbound_cost
        communication_cost = user_communication_inbound_cost + total_outbound_comm_cost
        if server_only_mode:
            api_cost = (max(0.0, float(total_draft_time_sec)) + max(0.0, float(target_billed_time_sec))) * max(
                0.0, float(target_per_sec_cost)
            )
        else:
            api_cost = target_cost + total_outbound_comm_cost
        server_only_wall_billed_time_sec = (
            max(0.0, float(total_time))
            if (server_only_mode and str(objective_metric).lower() == "total_cost")
            else 0.0
        )
        server_only_wall_total_cost = (
            server_only_wall_billed_time_sec * max(0.0, float(target_per_sec_cost))
            if (server_only_mode and str(objective_metric).lower() == "total_cost")
            else 0.0
        )
        if server_only_mode and str(objective_metric).lower() == "total_cost":
            # server-only total_cost: wall-time
            total_cost = server_only_wall_total_cost
        else:
            total_cost = draft_cost + target_cost + user_communication_inbound_cost + total_outbound_comm_cost
        
        # cost_per_token
        cost_per_token = total_cost / total_new_tokens if total_new_tokens > 0 else 0.0
        draft_cost_per_token = draft_cost / total_new_tokens if total_new_tokens > 0 else 0.0
        target_cost_per_token = target_cost / total_new_tokens if total_new_tokens > 0 else 0.0
        communication_cost_per_token = communication_cost / total_new_tokens if total_new_tokens > 0 else 0.0
        user_communication_inbound_cost_per_token = (
            user_communication_inbound_cost / total_new_tokens if total_new_tokens > 0 else 0.0
        )
        cloud_outbound_cost_per_token = cloud_outbound_cost / total_new_tokens if total_new_tokens > 0 else 0.0
        user_communication_outbound_cost_per_token = (
            user_communication_outbound_cost / total_new_tokens if total_new_tokens > 0 else 0.0
        )
        api_cost_per_token = api_cost / total_new_tokens if total_new_tokens > 0 else 0.0
        proactive_draft_cost_per_token = proactive_draft_cost / total_new_tokens if total_new_tokens > 0 else 0.0

        # 1M
        cost_per_1m_tokens = cost_per_token * 1_000_000.0
        draft_cost_per_1m_tokens = draft_cost_per_token * 1_000_000.0
        target_cost_per_1m_tokens = target_cost_per_token * 1_000_000.0
        communication_cost_per_1m_tokens = communication_cost_per_token * 1_000_000.0
        user_communication_inbound_cost_per_1m_tokens = (
            user_communication_inbound_cost_per_token * 1_000_000.0
        )
        cloud_outbound_cost_per_1m_tokens = cloud_outbound_cost_per_token * 1_000_000.0
        user_communication_outbound_cost_per_1m_tokens = (
            user_communication_outbound_cost_per_token * 1_000_000.0
        )
        api_cost_per_1m_tokens = api_cost_per_token * 1_000_000.0
        proactive_draft_cost_per_1m_tokens = proactive_draft_cost_per_token * 1_000_000.0

        # 1M (kWh / 1M tokens)
        draft_energy_kwh_per_token = (total_draft_energy_kwh / total_new_tokens) if total_new_tokens > 0 else 0.0
        draft_energy_kwh_per_1m_tokens = draft_energy_kwh_per_token * 1_000_000.0
        target_energy_kwh_per_token = (total_target_energy_kwh / total_new_tokens) if total_new_tokens > 0 else 0.0
        target_energy_kwh_per_1m_tokens = target_energy_kwh_per_token * 1_000_000.0
        total_energy_kwh = float(total_draft_energy_kwh + total_target_energy_kwh)
        total_energy_kwh_per_token = (total_energy_kwh / total_new_tokens) if total_new_tokens > 0 else 0.0
        total_energy_kwh_per_1m_tokens = total_energy_kwh_per_token * 1_000_000.0

        print(
            "total_steps: {}  | total_new_tokens: {}  | total_time: {:.2f}  | profile_update_overhead: {:.4f}  | tokens per step: {:.2f}  | cost per 1M token: {:.6f}  | tokens per second: {:.2f}  | avg_tree_depth: {:.2f}".format(
                grand_total_steps,
                total_new_tokens,
                total_time,
                float(profile_update_overhead_sec_total),
                tokens_per_step,
                cost_per_1m_tokens,
                tokens_per_second,
                avg_tree_depth,
            )
        )
        if total_draft_energy_kwh > 0:
            print(
                "draft energy total: {:.6f} kWh  |  draft energy per 1M token: {:.6f} kWh".format(
                    total_draft_energy_kwh,
                    draft_energy_kwh_per_1m_tokens,
                )
            )
        if total_target_energy_kwh > 0:
            print(
                "target energy total: {:.6f} kWh  |  target energy per 1M token: {:.6f} kWh  |  total energy per 1M token: {:.6f} kWh".format(
                    total_target_energy_kwh,
                    target_energy_kwh_per_1m_tokens,
                    total_energy_kwh_per_1m_tokens,
                )
            )
        if objective_selection_mode == "constraint":
            cstats = dict(runner.constraint_fallback_stats)
            print(
                "[Constraint Debug] width selected fallback/feasible: {}/{} | nnodes selected fallback/feasible: {}/{}".format(
                    int(cstats.get("width_selected_fallback", 0)),
                    int(cstats.get("width_selected_feasible", 0)),
                    int(cstats.get("nnodes_selected_fallback", 0)),
                    int(cstats.get("nnodes_selected_feasible", 0)),
                )
            )
        
        # total_algorithm_time total_algorithm_cost
        # algorithm_stats total_algorithm_time 
        total_algorithm_time = algorithm_stats.get("total_algorithm_time", 0.0) if algorithm_stats else 0.0
        if str(objective_metric).lower() in {"total_cost", "api_cost"} and total_draft_time_sec > 0:
            avg_draft_cost_per_sec = float(draft_cost) / max(1e-9, float(total_draft_time_sec))
            total_algorithm_cost = total_algorithm_time * avg_draft_cost_per_sec
        else:
            total_algorithm_cost = total_algorithm_time * (
                target_per_sec_cost if bool(getattr(runner, "bill_draft_as_target_gpu", False)) else draft_per_sec_cost
            )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        integrated_result = {
            "experiment_info": {
                "draft_model": draft_model_path,
                "base_model": base_model_path,
                "bench_name": bench_name,
                "device_name": device_name,
                "timestamp": timestamp,
                "experiment_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "nodes": nodes,
                "max_depth": max_depth,
                "temperature": temperature,
                "draft_per_sec_cost": draft_per_sec_cost,
                "draft_electricity_cost_per_kwh": draft_electricity_cost_per_kwh,
                "target_per_sec_cost": target_per_sec_cost,
                "user_communication_cost_per_gb": float(user_communication_cost_per_gb),
                "cloud_outbound_cost_per_gb": float(cloud_outbound_cost_per_gb),
                "accept_length_margin": float(accept_length_margin),
                "objective_selection_mode": str(objective_selection_mode),
                "constraint_target": str(constraint_target),
                "metric_constraint_per_1m_token": (
                    float(metric_constraint_per_1m_token)
                    if metric_constraint_per_1m_token is not None
                    else None
                ),
                "min_tps_constraint": (
                    float(min_tps_constraint)
                    if min_tps_constraint is not None
                    else 0.0
                ),
                "total_metric_cap": (
                    float(total_metric_cap)
                    if np.isfinite(total_metric_cap)
                    else "inf"
                ),
                "cost_sensitivity": cost_sensitivity,
                "objective_metric": objective_metric,
                "reference_tps": float(runner.reference_tps),
                "reference_cost_per_token": float(runner.reference_cost_per_token),
                "reference_objective_per_token": float(runner.reference_objective_per_token),
                "reference_feasible_metric_per_token": reference_feasible_metric_per_token,
                "reference_feasible_tps": reference_feasible_tps,
                "reference_cs_anchor_curve": reference_cs_anchor_curve,
                "reference_tradeoff_curve_cs0_1": reference_tradeoff_curve_cs0_1,
                "reference_constraint_anchor_curve": reference_constraint_anchor_curve,
                "reference_tradeoff_curve_by_constraint": reference_tradeoff_curve_by_constraint,
                "reference_constraint_center_per_1m_token": (
                    float(auto_reference_constraint_center_per_1m)
                    if auto_reference_constraint_center_per_1m is not None
                    else None
                ),
                "reference_cs_curve_rounds": int(reference_cs_curve_rounds),
                "reference_constraint_multipliers": [
                    float(v) for v in reference_constraint_multipliers_list
                ],
                "metric_constraint_source": (
                    "user"
                    if user_metric_constraint_provided
                    else ("auto_blend_cs50" if objective_selection_mode == "constraint" else None)
                ),
                "no_draft_cost": bool(no_draft_cost),
                "opt_tree": opt_tree,
                "fixed_width": fixed_width,
                "fixed_depth": fixed_depth,
                "fixed_nnodes": fixed_nnodes,
                "proactive_drafting": bool(proactive_drafting),
                "proactive_budget_enabled": bool(proactive_drafting and not disable_proactive_budget),
                "join_canceled_proactive_before_tree_build": bool(join_canceled_proactive_before_tree_build),
                "proactive_threshold": float(proactive_threshold),
                "adaptive_proactive_threshold": bool(adaptive_proactive_threshold),
                "server_only_mode": bool(server_only_mode),
                "disable_server_only": bool(disable_server_only),
                "gpu_monitor_enabled": gpu_monitor_enabled,
                "gpu_monitor_sample_count": int(gpu_monitor_sample_count),
                "gpu_monitor_backend": (
                    str(getattr(runner.gpu_monitor, "backend", "none"))
                    if runner.gpu_monitor is not None else "none"
                ),
                "gpu_monitor_long_running": bool(gpu_monitor_long_running),
                "gpu_monitor_power_draw_available": gpu_monitor_power_draw_available,
                "gpu_monitor_status": gpu_monitor_status,
                # ``args`` is only defined in module scope when this script is
                # run via ``python -m``. When run_draft is called as a Python
                # function (e.g. from autodraft.local_runner), ``args`` does
                # not exist; fall back via globals().get to avoid NameError.
                "seed": getattr(globals().get("args"), "seed", None),
                "deterministic": bool(getattr(globals().get("args"), "deterministic", False)),
                "online_profile_update": bool(online_profile_update),
                "online_profile_lr": float(_sanitize_online_lr(online_profile_lr)),
                "accept_length_calibration_enabled": bool(accept_length_calibration),
                "target_time_calibration_enabled": bool(target_time_calibration),
                "network_info": network_info,
                "gpu_info": gpu_info,
            },
            "generation_stats": {
                "total_steps": grand_total_steps,
                "total_new_tokens": total_new_tokens,
                "total_time_seconds": total_time,
                "effective_time_seconds": float(total_time),
                "profile_update_overhead_seconds": float(profile_update_overhead_sec_total),
                "tokens_per_step": tokens_per_step,
                "tokens_per_second": tokens_per_second,
                "tokens_per_second_effective": float(tokens_per_second_wall),
                "tokens_per_second_wall": float(tokens_per_second_wall),
                "avg_tree_depth": avg_tree_depth,
                "avg_tree_width": avg_tree_width,
                "avg_final_nodes": avg_final_nodes,
                "avg_width_per_depth": depth_avg_widths,
                "avg_accept_length": avg_accept_length,
                "original_expected_accept_length": avg_expected_accept_length,
                "scaled_expected_accept_length": avg_scaled_expected_accept_length,
                "scaled_with_margin_expected_accept_length": (
                    float(avg_scaled_expected_accept_length * (1.0 - float(getattr(runner, "accept_length_margin", 0.05))))
                    if avg_scaled_expected_accept_length is not None
                    else None
                ),
                "clipped_expected_accept_length": avg_clipped_expected_accept_length,
                "avg_accept_length_scale": runner.get_accept_length_ratio_mean(),
                "accept_length_calibration_enabled": bool(accept_length_calibration),
                "target_time_calibration_enabled": bool(target_time_calibration),
                "avg_expected_accept_length": avg_expected_accept_length,
                "avg_accept_length_per_expected": avg_accept_length_per_expected,
                "proactive_draft_attempts": proactive_draft_attempts,
                "proactive_tree_used": proactive_tree_used,
                "proactive_budget_enabled": bool(proactive_drafting and not disable_proactive_budget),
                "join_canceled_proactive_before_tree_build": bool(join_canceled_proactive_before_tree_build),
                "proactive_start_skipped_by_value_count": int(proactive_start_skipped_by_value_count),
                "proactive_start_expected_gain_total_ms": float(proactive_start_expected_gain_ms_sum),
                "proactive_start_expected_cancel_loss_total_ms": float(proactive_start_expected_cancel_loss_ms_sum),
                "proactive_start_value_score_avg": (
                    float(proactive_start_value_score_sum / proactive_start_value_count)
                    if proactive_start_value_count > 0 else 0.0
                ),
                "proactive_start_value_count": int(proactive_start_value_count),
                "proactive_start_cancel_loss_estimate_latest_ms": (
                    float(proactive_start_cancel_loss_estimate_ms_latest)
                    if proactive_start_cancel_loss_estimate_ms_latest is not None
                    else None
                ),
                "proactive_path_match_count": int(proactive_path_match_count),
                "proactive_path_mismatch_count": int(proactive_path_mismatch_count),
                "proactive_worker_reuse_enabled": bool(proactive_worker is not None),
                "proactive_worker_tasks_submitted": int(proactive_worker.submitted) if proactive_worker is not None else 0,
                "proactive_worker_tasks_completed": int(proactive_worker.completed) if proactive_worker is not None else 0,
                "recv_worker_reuse_enabled": bool(recv_worker is not None),
                "recv_worker_tasks_submitted": int(recv_worker.submitted) if recv_worker is not None else 0,
                "recv_worker_tasks_completed": int(recv_worker.completed) if recv_worker is not None else 0,
                "proactive_wall_elapsed_seconds": float(proactive_wall_elapsed_total_sec),
                "proactive_compute_elapsed_seconds": float(proactive_total_time_sec),
                "proactive_compute_used_seconds": float(proactive_compute_used_sec),
                "proactive_compute_unused_seconds": float(proactive_compute_unused_sec),
                "proactive_compute_hidden_before_reply_used_seconds": float(
                    proactive_compute_hidden_used_sec
                ),
                "proactive_compute_after_reply_used_seconds": float(
                    proactive_compute_after_reply_used_sec
                ),
                "proactive_effective_hidden_gain_seconds": float(
                    proactive_compute_hidden_used_sec
                ),
                "proactive_effective_net_gain_seconds": float(
                    proactive_compute_hidden_used_sec
                    - (float(proactive_cancel_to_exit_ms_sum) / 1000.0)
                ),
                "proactive_net_gain_seconds": float(
                    max(0.0, float(proactive_total_time_sec))
                    - (float(proactive_cancel_to_exit_ms_sum) / 1000.0)
                ),
                "proactive_budget_latest_ms": (
                    float(proactive_budget_ms_latest)
                    if proactive_budget_ms_latest is not None
                    else None
                ),
                "proactive_budget_wait_total_ms": float(proactive_budget_wait_ms_sum),
                "proactive_budget_wait_avg_ms": (
                    float(proactive_budget_wait_ms_sum / proactive_budget_wait_ms_count)
                    if proactive_budget_wait_ms_count > 0 else 0.0
                ),
                "proactive_budget_wait_max_ms": float(proactive_budget_wait_ms_max),
                "proactive_budget_wait_count": int(proactive_budget_wait_ms_count),
                "proactive_resume_after_reply_total_ms": float(proactive_resume_after_reply_ms_sum),
                "proactive_resume_after_reply_avg_ms": (
                    float(proactive_resume_after_reply_ms_sum / proactive_resume_after_reply_ms_count)
                    if proactive_resume_after_reply_ms_count > 0 else 0.0
                ),
                "proactive_resume_after_reply_max_ms": float(proactive_resume_after_reply_ms_max),
                "proactive_resume_after_reply_count": int(proactive_resume_after_reply_ms_count),
                "proactive_post_reply_wait_total_ms": float(proactive_post_reply_wait_ms_sum),
                "proactive_post_reply_wait_avg_ms": (
                    float(proactive_post_reply_wait_ms_sum / proactive_post_reply_wait_ms_count)
                    if proactive_post_reply_wait_ms_count > 0 else 0.0
                ),
                "proactive_post_reply_wait_max_ms": float(proactive_post_reply_wait_ms_max),
                "proactive_post_reply_wait_count": int(proactive_post_reply_wait_ms_count),
                "proactive_cancel_to_exit_total_ms": float(proactive_cancel_to_exit_ms_sum),
                "proactive_cancel_to_exit_avg_ms": (
                    float(proactive_cancel_to_exit_ms_sum / proactive_cancel_to_exit_ms_count)
                    if proactive_cancel_to_exit_ms_count > 0 else 0.0
                ),
                "proactive_cancel_to_exit_max_ms": float(proactive_cancel_to_exit_ms_max),
                "proactive_cancel_to_exit_count": int(proactive_cancel_to_exit_ms_count),
                "proactive_cancel_pending_count": int(proactive_cancel_pending_count),
                "proactive_cancel_immediate_join_total_ms": float(proactive_cancel_immediate_join_ms_sum),
                "proactive_cancel_immediate_join_avg_ms": (
                    float(proactive_cancel_immediate_join_ms_sum / proactive_cancel_immediate_join_ms_count)
                    if proactive_cancel_immediate_join_ms_count > 0 else 0.0
                ),
                "proactive_cancel_immediate_join_max_ms": float(proactive_cancel_immediate_join_ms_max),
                "proactive_cancel_immediate_join_count": int(proactive_cancel_immediate_join_ms_count),
                "proactive_cancel_join_before_tree_build_total_ms": float(proactive_cancel_join_before_tree_build_ms_sum),
                "proactive_cancel_join_before_tree_build_avg_ms": (
                    float(proactive_cancel_join_before_tree_build_ms_sum / proactive_cancel_join_before_tree_build_ms_count)
                    if proactive_cancel_join_before_tree_build_ms_count > 0 else 0.0
                ),
                "proactive_cancel_join_before_tree_build_max_ms": float(proactive_cancel_join_before_tree_build_ms_max),
                "proactive_cancel_join_before_tree_build_count": int(proactive_cancel_join_before_tree_build_ms_count),
                "proactive_cancel_tree_build_overlap_total_ms": float(proactive_cancel_tree_build_overlap_ms_sum),
                "proactive_cancel_tree_build_overlap_avg_ms": (
                    float(proactive_cancel_tree_build_overlap_ms_sum / proactive_cancel_tree_build_overlap_ms_count)
                    if proactive_cancel_tree_build_overlap_ms_count > 0 else 0.0
                ),
                "proactive_cancel_tree_build_overlap_max_ms": float(proactive_cancel_tree_build_overlap_ms_max),
                "proactive_cancel_tree_build_overlap_count": int(proactive_cancel_tree_build_overlap_ms_count),
                "proactive_cancel_tree_build_overlap_event_count": int(proactive_cancel_tree_build_overlap_event_count),
                "proactive_finalize_early_count": int(proactive_finalize_early_count),
                "proactive_expand_continue_count": int(proactive_expand_continue_count),
                "proactive_expand_pause_count": int(proactive_expand_pause_count),
                "proactive_expected_gain_total_ms": float(proactive_expected_gain_ms_sum),
                "proactive_expected_loss_total_ms": float(proactive_expected_loss_ms_sum),
                "proactive_depth_stats": {
                    str(k): {"used": int(v.get("used", 0) or 0), "canceled": int(v.get("canceled", 0) or 0)}
                    for k, v in sorted(proactive_depth_stats.items())
                },
                "proactive_post_reply_by_depth": _finalize_post_reply_bucket(proactive_post_reply_by_depth),
                "proactive_post_reply_by_width": _finalize_post_reply_bucket(proactive_post_reply_by_width),
                "proactive_post_reply_by_depth_width": _finalize_post_reply_bucket(proactive_post_reply_by_depth_width),
                "proactive_expand_depth_counts": {
                    str(k): int(v) for k, v in sorted(proactive_expand_depth_counts.items())
                },
                "proactive_used_path_prob_avg": (
                    float(proactive_used_path_prob_sum / proactive_used_path_prob_count)
                    if proactive_used_path_prob_count > 0 else None
                ),
                "proactive_unused_path_prob_avg": (
                    float(proactive_unused_path_prob_sum / proactive_unused_path_prob_count)
                    if proactive_unused_path_prob_count > 0 else None
                ),
                "proactive_used_path_prob_count": proactive_used_path_prob_count,
                "proactive_unused_path_prob_count": proactive_unused_path_prob_count,
                "proactive_used_avg_tree_depth": (
                    float(proactive_used_tree_depth_sum / proactive_used_tree_steps)
                    if proactive_used_tree_steps > 0 else None
                ),
                "proactive_used_avg_tree_width": (
                    float(proactive_used_tree_width_sum / proactive_used_tree_steps)
                    if proactive_used_tree_steps > 0 else None
                ),
                "proactive_used_avg_final_nodes": (
                    float(proactive_used_tree_final_nodes_sum / proactive_used_tree_steps)
                    if proactive_used_tree_steps > 0 else None
                ),
                "proactive_used_avg_accept_length": (
                    float(proactive_used_tree_accept_length_sum / proactive_used_tree_steps)
                    if proactive_used_tree_steps > 0 else None
                ),
                "proactive_used_avg_scaled_expected_accept_length": (
                    float(proactive_used_tree_scaled_expected_sum / proactive_used_tree_steps)
                    if proactive_used_tree_steps > 0 else None
                ),
                "proactive_used_steps": proactive_used_tree_steps,
                "foreground_tree_steps": int(foreground_tree_steps),
                "foreground_tree_avg_accept_length": (
                    float(foreground_tree_accept_length_sum / foreground_tree_steps)
                    if foreground_tree_steps > 0 else None
                ),
                "after_proactive_foreground_steps": int(after_proactive_foreground_steps),
                "after_proactive_foreground_avg_accept_length": (
                    float(after_proactive_foreground_accept_length_sum / after_proactive_foreground_steps)
                    if after_proactive_foreground_steps > 0 else None
                ),
                "after_proactive_cancel_foreground_steps": int(after_proactive_cancel_foreground_steps),
                "after_proactive_cancel_foreground_avg_accept_length": (
                    float(after_proactive_cancel_foreground_accept_length_sum / after_proactive_cancel_foreground_steps)
                    if after_proactive_cancel_foreground_steps > 0 else None
                ),
                # Average draft-time ratio from a profiling run
                "avg_draft_time_ratio": runner.get_draft_time_ratio_mean(),
                "avg_target_verification_time_ratio": runner.get_target_verification_ratio_mean(),
                "acceptance_ratio_avg": reported_acceptance_ratio_avg,
                "total_cost": float(total_cost),
                "api_cost": float(api_cost),
                "draft_cost": float(draft_cost),
                "target_cost": float(target_cost),
                "draft_cost_source": (
                    "server_wall_time_x_target_per_sec_cost"
                    if (server_only_mode and str(objective_metric).lower() == "total_cost")
                    else (
                    "time_x_target_per_sec_cost"
                    if (
                        (not bool(no_draft_cost))
                        and (
                            bool(getattr(runner, "bill_draft_as_target_gpu", False))
                            or (server_only_mode and str(objective_metric).lower() == "api_cost")
                        )
                    )
                    else (
                        "gpu_energy_kwh_x_electricity_price"
                        if (str(objective_metric).lower() == "total_cost" and not bool(no_draft_cost))
                        else "time_x_draft_per_sec_cost"
                    )
                    )
                ),
                "user_communication_inbound_cost": float(user_communication_inbound_cost),
                "cloud_outbound_cost": float(cloud_outbound_cost),
                "user_communication_outbound_cost": float(user_communication_outbound_cost),
                "communication_cost": float(communication_cost),
                "total_transfer_bytes": float(total_transfer_bytes),
                "cost_per_1m_tokens": float(cost_per_1m_tokens),
                "total_cost_per_1m_tokens": float(cost_per_1m_tokens),
                "api_cost_per_1m_tokens": float(api_cost_per_1m_tokens),
                "draft_cost_per_1m_tokens": float(draft_cost_per_1m_tokens),
                "target_cost_per_1m_tokens": float(target_cost_per_1m_tokens),
                "user_communication_inbound_cost_per_1m_tokens": float(user_communication_inbound_cost_per_1m_tokens),
                "cloud_outbound_cost_per_1m_tokens": float(cloud_outbound_cost_per_1m_tokens),
                "user_communication_outbound_cost_per_1m_tokens": float(user_communication_outbound_cost_per_1m_tokens),
                "communication_cost_per_1m_tokens": float(communication_cost_per_1m_tokens),
                "proactive_draft_cost_per_1m_tokens": float(proactive_draft_cost_per_1m_tokens),
                "bill_draft_as_target_gpu": bool(getattr(runner, "bill_draft_as_target_gpu", False)),
                "server_only_total_session_time_seconds": float(server_only_total_session_time_sec),
                "server_only_profile_prepare_overhead_seconds": float(server_only_profile_prepare_overhead_sec),
                "server_only_transfer_time_seconds": float(server_only_transfer_time_sec),
                "server_only_wall_billed_time_seconds": float(server_only_wall_billed_time_sec),
                "server_only_wall_total_cost": float(server_only_wall_total_cost),
                "target_billed_time_seconds": float(target_billed_time_sec),
                "total_draft_time_seconds": float(total_draft_time_sec),
                "total_tree_build_wall_time_seconds": float(total_tree_build_wall_time_sec),
                "proactive_draft_time_seconds": float(proactive_total_time_sec),
                "proactive_draft_cost": float(proactive_draft_cost),
                "foreground_draft_gpu_energy_kwh": float(foreground_draft_gpu_energy_kwh),
                "proactive_draft_gpu_energy_kwh": float(proactive_draft_gpu_energy_kwh),
                "proactive_draft_gpu_used_energy_kwh": float(proactive_draft_gpu_used_energy_kwh),
                "proactive_draft_gpu_unused_energy_kwh": float(proactive_draft_gpu_unused_energy_kwh),
                "proactive_gpu_sample_count": int(len(proactive_gpu_data)),
                "total_draft_gpu_energy_kwh": float(total_draft_gpu_energy_kwh),
                "total_draft_cpu_energy_kwh": float(total_draft_cpu_energy_kwh),
                "total_draft_energy_kwh": float(total_draft_energy_kwh),
                "draft_energy_kwh_per_1m_tokens": float(draft_energy_kwh_per_1m_tokens),
                "total_target_energy_kwh": float(total_target_energy_kwh),
                "target_energy_kwh_per_1m_tokens": float(target_energy_kwh_per_1m_tokens),
                "target_energy_sample_count": int(target_energy_sample_count),
                "target_energy_missing_count": int(target_energy_missing_count),
                "total_energy_kwh": float(total_energy_kwh),
                "total_energy_kwh_per_1m_tokens": float(total_energy_kwh_per_1m_tokens),
                "all_objective_metrics_per_1m": {
                    "total_cost": float(cost_per_1m_tokens),
                    "api_cost": float(api_cost_per_1m_tokens),
                    "draft_energy": float(draft_energy_kwh_per_1m_tokens),
                    "target_energy": float(target_energy_kwh_per_1m_tokens),
                },
                "cost_per_token": float(cost_per_token),
                "api_cost_per_token": float(api_cost_per_token),
                "communication_cost_per_token": float(communication_cost_per_token),
                "user_communication_inbound_cost_per_token": float(user_communication_inbound_cost_per_token),
                "cloud_outbound_cost_per_token": float(cloud_outbound_cost_per_token),
                "user_communication_outbound_cost_per_token": float(user_communication_outbound_cost_per_token),
                "metric_spent_total": float(metric_spent_total),
                "metric_cap_reached": bool(metric_cap_reached),
                "target_profile_lookup_stats": dict(runner.target_profile_lookup_stats),
                "constraint_fallback_stats": dict(runner.constraint_fallback_stats),
                "total_algorithm_time": float(total_algorithm_time),
                "total_algorithm_cost": float(total_algorithm_cost),
            },
            "network_latency": {
                "global": {
                    "avg_draft_to_target_time_ms": global_avg_draft,
                    "avg_target_to_draft_time_ms": global_avg_target,
                    "avg_accept_length": global_avg_accept,
                    "total_steps": grand_total_steps,
                    "accept_length_bucket_stats": global_accept_len_bucket_stats,
                    "final_nnodes_bucket_stats": global_final_nnodes_bucket_stats,
                },
                "questions": network_latency_records,
            },
            "reference_update_history": reference_update_history,
            "depth_statistics": depth_statistics,
            "depth_statistics_short": depth_statistics_short,
            "algorithm_stats": algorithm_stats,  # Algorithm (objective_value, per_token_latency, per_token_cost )
            "width_timing_stats": width_timing_stats,  # Width draft_model.model
            "model_function_timing": model_function_timing,  # model.model()
            "latency_statistics": latency_statistics,  # latency (raw + stats)
            "data_transfer_stats": data_transfer_stats,  # (KByte )
            "accept_stats": accept_stats,  # Accept length (target )
            "accept_length_step_pairs": accept_length_step_pairs,  # step predicted-vs-actual accept length
            "tree_width_records": tree_width_records,  # depth width, width, final_nnodes
            "draft_gpu_summary": draft_gpu_summary if draft_gpu_summary else {},  # Draft GPU
            "proactive_draft_gpu_summary": proactive_draft_gpu_summary if proactive_draft_gpu_summary else {},
            "draft_cpu_power_summary": draft_cpu_power_summary if draft_cpu_power_summary else {},  # Draft CPU
            "answers": all_answers,
        }
        
        with open(output_file, 'w') as f:
            json.dump(integrated_result, f, indent=2)
        
        # Trimmed (raw )
        def create_trimmed_version(data):
            """Create a trimmed version by removing raw values and all_values fields."""
            import copy
            trimmed = copy.deepcopy(data)
            
            # latency_statistics raw
            if "latency_statistics" in trimmed:
                for key in [
                    "tree_build",
                    "draft_to_target",
                    "target_verification",
                    "target_to_draft",
                    "tree_model_forward",
                    "tree_width_algo",
                    "tree_nnodes_algo",
                    "tree_mask_build",
                    "tree_finalize",
                ]:
                    if key in trimmed["latency_statistics"]:
                        if "raw" in trimmed["latency_statistics"][key]:
                            del trimmed["latency_statistics"][key]["raw"]
            
            # depth_statistics all_values
            if "depth_statistics" in trimmed:
                for depth, stats in trimmed["depth_statistics"].items():
                    for key in stats:
                        if isinstance(stats[key], dict) and "all_values" in stats[key]:
                            del stats[key]["all_values"]
            
            # width_timing_stats all_times_ms
            if "width_timing_stats" in trimmed:
                for width, stats in trimmed["width_timing_stats"].items():
                    if "all_times_ms" in stats:
                        del stats["all_times_ms"]
            
            # data_transfer_stats raw_kbytes
            if "data_transfer_stats" in trimmed:
                for key in ["draft_to_target", "target_to_draft"]:
                    if key in trimmed["data_transfer_stats"]:
                        if "raw_kbytes" in trimmed["data_transfer_stats"][key]:
                            del trimmed["data_transfer_stats"][key]["raw_kbytes"]
            
            # network_latency global accept_length_bucket_stats final_nnodes_bucket_stats times_ms
            if "network_latency" in trimmed and "global" in trimmed["network_latency"]:
                if "accept_length_bucket_stats" in trimmed["network_latency"]["global"]:
                    for length, stats in trimmed["network_latency"]["global"]["accept_length_bucket_stats"].items():
                        if "times_ms" in stats:
                            del stats["times_ms"]
                if "final_nnodes_bucket_stats" in trimmed["network_latency"]["global"]:
                    for nnodes, stats in trimmed["network_latency"]["global"]["final_nnodes_bucket_stats"].items():
                        if "times_ms" in stats:
                            del stats["times_ms"]
            
            # network_latency questions accept_length_bucket_stats final_nnodes_bucket_stats times_ms
            if "network_latency" in trimmed and "questions" in trimmed["network_latency"]:
                for question_record in trimmed["network_latency"]["questions"]:
                    if "accept_length_bucket_stats" in question_record:
                        for length, stats in question_record["accept_length_bucket_stats"].items():
                            if "times_ms" in stats:
                                del stats["times_ms"]
                    if "final_nnodes_bucket_stats" in question_record:
                        for nnodes, stats in question_record["final_nnodes_bucket_stats"].items():
                            if "times_ms" in stats:
                                del stats["times_ms"]
            
            return trimmed
        
        # Trimmed
        trimmed_result = create_trimmed_version(integrated_result)
        
        # Trimmed ( _trimmed )
        base_name = os.path.splitext(output_file)[0]
        file_ext = os.path.splitext(output_file)[1]
        trimmed_output_file = f"{base_name}_trimmed{file_ext}"
        
        # Trimmed
        with open(trimmed_output_file, 'w') as f:
            json.dump(trimmed_result, f, indent=2)
        
        print(f"\nCombined result file saved:")
        print(f"  - Original: {output_file}")
        print(f"  - Trimmed: {trimmed_output_file}")
        print(f"  - Answer count: {len(all_answers)}")
        print(f"  - Network latency records: {len(network_latency_records)} questions")
        print(f"  - Statistics by depth: {len(depth_statistics)}depths")


def run_chat_mode(
    host: str,
    port: int,
    base_model_path: str,
    draft_model_path: str,
    bench_name: str,
    question_file: str,
    temperature: float,
    nodes: int,
    max_depth: int,
    device_map: str,
    load_in_4bit: bool,
    load_in_8bit: bool,
    target_quantization: str,
    per_token_probability_bound: float,
    per_path_probability_bound: float,
    device_name: str,
    draft_per_hour_cost: float,
    target_per_hour_cost: float,
    draft_electricity_cost_per_kwh: float,
    user_communication_cost_per_gb: float,
    cloud_outbound_cost_per_gb: float,
    accept_length_margin: float,
    objective_selection_mode: str,
    constraint_target: str,
    metric_constraint_per_1m_token: float,
    min_tps_constraint: float,
    cost_sensitivity: float,
    min_width: int,
    fixed_depth: bool,
    fixed_nnodes: bool,
    fixed_width: bool,
    fixed_width_value: int,
    server_name: str,
    enable_gpu_monitor: bool,
    gpu_monitor_interval: float,
    enable_cpu_monitor: bool,
    objective_metric: str,
    no_draft_cost: bool,
    fix_gpu_clock: bool = False,
    gpu_graphics_clock_mhz: int = None,
    gpu_memory_clock_mhz: int = None,
    tokenizer_path: str = None,
    chat_max_new_tokens: int = 512,
    auto_target_profile: bool = True,
    draft_profile_force_refresh: bool = False,
    draft_profile_width_list: str = "10,20,30,40,50,60,70,80,90,100,110,120,130,140,150",
    target_profile_force_refresh: bool = False,
    target_profile_model_calls_per_count: int = 20,
    target_profile_width_list: str = "50",
    target_profile_depth_list: str = "10",
    target_profile_node_list: str = "10,20,30,40,50,60,70,80,90,100,110,120,130,140,150",
    online_profile_update: bool = True,
    online_profile_lr: float = ONLINE_PROFILE_LR_DEFAULT,
    bill_draft_as_target_gpu: bool = False,
    server_draft_profile_auto: bool = True,
    server_draft_profile_force_refresh: bool = False,
    server_draft_profile_model_calls_per_count: int = 100,
    server_draft_profile_width_list: str = "10,20,30,40,50,60,70,80,90,100,110,120,130,140,150",
    **_unused_kwargs,
):
    """Persistent multi-turn chat mode using hybrid speculative decoding."""
    requested_draft_quantization = (
        "8bit" if bool(load_in_8bit) else ("4bit" if bool(load_in_4bit) else "none")
    )
    draft_quantization = requested_draft_quantization
    objective_metric = _normalize_objective_metric(objective_metric)
    if objective_metric == "total_cost" and (not no_draft_cost) and (not bill_draft_as_target_gpu) and (not enable_gpu_monitor):
        print("[INFO] chat_mode total_cost -> enabling GPU monitor (required for GPU-energy-based draft cost).")
        enable_gpu_monitor = True
    if objective_metric == "total_cost":
        if float(draft_per_hour_cost or 0.0) > 0:
            print(
                "[WARN] --draft-per-hour-cost is ignored for objective_metric=total_cost. "
                "Use --draft-electricity-cost-per-kwh instead."
            )
        draft_per_sec_cost = 0.0
    else:
        draft_per_sec_cost = float(draft_per_hour_cost) / 3600.0
    target_per_sec_cost = float(target_per_hour_cost) / 3600.0
    if bool(bill_draft_as_target_gpu) and (not bool(no_draft_cost)):
        draft_per_sec_cost = float(target_per_sec_cost)
    metric_constraint_per_token = (
        (float(metric_constraint_per_1m_token) / 1_000_000.0)
        if str(constraint_target).lower() == "metric" and metric_constraint_per_1m_token is not None
        else None
    )
    objective_selection_mode = str(objective_selection_mode).lower()
    constraint_target = str(constraint_target).lower()
    user_communication_cost_per_gb, cloud_outbound_cost_per_gb = _resolve_cloud_transfer_costs(
        user_communication_cost_per_gb=user_communication_cost_per_gb,
        cloud_outbound_cost_per_gb=cloud_outbound_cost_per_gb,
    )

    KVCls = get_kv_llama_class(base_model_path)
    draft_quant_fallback_chain = _build_draft_quantization_fallback_chain(
        requested_draft_quantization,
        load_in_4bit=bool(load_in_4bit),
        load_in_8bit=bool(load_in_8bit),
    )
    draft_model = None
    attempted_draft_quants: List[str] = []
    draft_failures: List[str] = []
    for quant_mode in draft_quant_fallback_chain:
        attempted_draft_quants.append(str(quant_mode))
        quantization_config = _build_quantization_config_for_mode(quant_mode)
        try:
            draft_model = KVCls.from_pretrained(
                draft_model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map=device_map,
                quantization_config=quantization_config,
                token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None,
            )
            draft_quantization = str(quant_mode)
            break
        except Exception as e:
            draft_failures.append(f"{quant_mode}:{e}")
            _release_partial_draft_model(draft_model)
            draft_model = None
            if not _is_memory_related_load_error(str(e)):
                break
    if draft_model is None:
        failures_text = " | ".join(draft_failures) if draft_failures else "unknown_failure"
        attempted_text = " -> ".join(attempted_draft_quants) if attempted_draft_quants else "none"
        raise RuntimeError(
            "Failed to load draft model after quantization fallback. "
            f"attempted={attempted_text}; failures={failures_text}. "
            "Try a smaller draft model or start with higher quantization."
        )
    if len(attempted_draft_quants) > 1 or draft_quantization != requested_draft_quantization:
        print(
            "[Draft Fallback] loaded with quantization="
            f"{draft_quantization} (requested={requested_draft_quantization}, "
            f"attempted={attempted_draft_quants})"
        )
    tokenizer_source = tokenizer_path or base_model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        use_fast=False,
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None,
    )
    _ = prepare_logits_processor(temperature=temperature)
    draft_model.eval()

    runner = DraftRunner(
        draft_model=draft_model,
        tokenizer=tokenizer,
        debug=False,
        profile_data=None,
        draft_per_sec_cost=draft_per_sec_cost,
        target_per_sec_cost=target_per_sec_cost,
        draft_electricity_cost_per_kwh=draft_electricity_cost_per_kwh,
        user_communication_cost_per_gb=user_communication_cost_per_gb,
        cloud_outbound_cost_per_gb=cloud_outbound_cost_per_gb,
        cost_sensitivity=cost_sensitivity,
        enable_gpu_monitor=enable_gpu_monitor,
        gpu_monitor_interval=gpu_monitor_interval,
        enable_cpu_monitor=enable_cpu_monitor,
        fix_gpu_clock=bool(fix_gpu_clock),
        gpu_graphics_clock_mhz=gpu_graphics_clock_mhz,
        gpu_memory_clock_mhz=gpu_memory_clock_mhz,
        opt_tree=False,
        no_draft_cost=no_draft_cost,
        objective_metric=objective_metric,
        accept_length_margin=accept_length_margin,
        objective_selection_mode=objective_selection_mode,
        constraint_target=constraint_target,
        metric_constraint_per_token=metric_constraint_per_token,
        min_tps_constraint=min_tps_constraint,
        bill_draft_as_target_gpu=bool(bill_draft_as_target_gpu),
    )
    if objective_metric == "total_cost" and (not no_draft_cost) and (not bill_draft_as_target_gpu):
        if runner.gpu_monitor is None:
            raise RuntimeError(
                "total_cost requires GPU monitor for draft energy cost, but GPU monitor is not initialized."
            )
        try:
            gpu_info = runner.gpu_monitor.get_gpu_info()
        except Exception as e:
            raise RuntimeError(
                f"total_cost requires GPU monitor, but failed to query GPU info: {e}"
            ) from e
        if not gpu_info:
            raise RuntimeError(
                "total_cost requires GPU monitor, but no GPU device info is available."
            )
    load_profile_data(
        runner=runner,
        tokenizer=tokenizer,
        max_depth=max_depth,
        draft_model_path=draft_model_path,
        base_model_path=base_model_path,
        bench_name=bench_name,
        device_name=device_name,
        server_name=server_name,
        target_quantization=target_quantization,
        draft_quantization=draft_quantization,
        question_file=question_file,
        fixed_depth=fixed_depth,
        debug=False,
        auto_target_profile=bool(auto_target_profile),
        draft_profile_force_refresh=bool(draft_profile_force_refresh),
        online_profile_update=bool(online_profile_update),
        online_profile_lr=float(online_profile_lr),
    )

    feasible_constraint_range_per_1m = None
    feasible_tps_range = None
    reference_tradeoff_curve_cs0_1 = None
    reference_tradeoff_curve_by_constraint = None
    reference_cs_anchor_curve = None
    reference_constraint_anchor_curve = None
    try:
        default_reference_constraint_multipliers = "0.8,1.0,1.2"
        reference_constraint_multipliers_list = _parse_reference_constraint_multipliers(
            default_reference_constraint_multipliers
        )
        if objective_selection_mode == "constraint":
            reference_mode_key = (
                f"{objective_selection_mode}|{constraint_target}|{objective_metric}|auto-center-blendcs50|"
                + ",".join(f"{v:.6f}" for v in reference_constraint_multipliers_list)
            )
        else:
            reference_mode_key = (
                f"{objective_selection_mode}|{constraint_target}|{objective_metric}|"
                + ",".join(f"{v:.6f}" for v in reference_constraint_multipliers_list)
            )
        reference_cache, _ = load_reference_cache(
            base_model_path=base_model_path,
            draft_model_path=draft_model_path,
            bench_name=bench_name,
            objective_metric=objective_metric,
            server_name=server_name,
            device_name=device_name,
            target_quantization=target_quantization,
            draft_quantization=draft_quantization,
            objective_selection_mode=objective_selection_mode,
            reference_mode_key=reference_mode_key,
        )
        if reference_cache and isinstance(reference_cache.get("feasible_metric_per_token", None), dict):
            f_min = reference_cache["feasible_metric_per_token"].get("min", None)
            f_max = reference_cache["feasible_metric_per_token"].get("max", None)
            if f_min is not None and f_max is not None:
                feasible_constraint_range_per_1m = {
                    "min": float(f_min) * 1_000_000.0,
                    "max": float(f_max) * 1_000_000.0,
                }
        if reference_cache and isinstance(reference_cache.get("feasible_tps", None), dict):
            t_min = reference_cache["feasible_tps"].get("min", None)
            t_max = reference_cache["feasible_tps"].get("max", None)
            if t_min is not None and t_max is not None:
                feasible_tps_range = {
                    "min": float(t_min),
                    "max": float(t_max),
                }
        if reference_cache and isinstance(reference_cache.get("reference_tradeoff_curve_cs0_1", None), list):
            reference_tradeoff_curve_cs0_1 = reference_cache.get("reference_tradeoff_curve_cs0_1")
        if reference_cache and isinstance(reference_cache.get("reference_tradeoff_curve_by_constraint", None), list):
            reference_tradeoff_curve_by_constraint = reference_cache.get("reference_tradeoff_curve_by_constraint")
        if reference_cache and isinstance(reference_cache.get("reference_cs_anchor_curve", None), list):
            reference_cs_anchor_curve = reference_cache.get("reference_cs_anchor_curve")
        if reference_cache and isinstance(reference_cache.get("reference_constraint_anchor_curve", None), list):
            reference_constraint_anchor_curve = reference_cache.get("reference_constraint_anchor_curve")
    except Exception:
        feasible_constraint_range_per_1m = None
        feasible_tps_range = None
        reference_tradeoff_curve_cs0_1 = None
        reference_tradeoff_curve_by_constraint = None
        reference_cs_anchor_curve = None
        reference_constraint_anchor_curve = None

    conv = _build_conversation_template_for_model(base_model_path)

    def _decode_output(tokens: List[int]) -> str:
        decoded = tokenizer.decode(tokens, spaces_between_special_tokens=False)
        if conv.stop_str and decoded.find(conv.stop_str) > 0:
            decoded = decoded[:decoded.find(conv.stop_str)]
        for special_token in tokenizer.special_tokens_map.values():
            if isinstance(special_token, list):
                for special_tok in special_token:
                    decoded = decoded.replace(special_tok, "")
            else:
                decoded = decoded.replace(special_token, "")
        for special_tok in (getattr(tokenizer, "all_special_tokens", []) or []):
            decoded = decoded.replace(str(special_tok), "")
        if conv.name == "xgen" and decoded.startswith("Assistant:"):
            decoded = decoded.replace("Assistant:", "", 1).strip()
        return decoded.strip()

    def _emit(obj: dict):
        print("CHAT_JSON:" + json.dumps(obj, ensure_ascii=False), flush=True)

    def _extract_total_gpu_power_w(gpu_stats: dict) -> float:
        if not isinstance(gpu_stats, dict):
            return 0.0
        vals = []
        for entry in gpu_stats.values():
            if not isinstance(entry, dict):
                continue
            pinfo = entry.get("power_draw_w")
            if isinstance(pinfo, dict):
                v = pinfo.get("avg", None)
                if v is not None:
                    try:
                        vals.append(float(v))
                    except Exception:
                        pass
        if not vals:
            return 0.0
        # Final results use total draft energy across GPUs; use summed power.
        return float(sum(vals))

    with _connect_target_with_retry(
        host=host,
        port=port,
        max_attempts=int(os.environ.get("AUTODRAFT_DRAFT_CONNECT_RETRIES", "8")),
        base_delay_s=float(os.environ.get("AUTODRAFT_DRAFT_CONNECT_BACKOFF_SEC", "0.5")),
        connect_timeout_s=float(os.environ.get("AUTODRAFT_DRAFT_CONNECT_TIMEOUT_SEC", "10.0")),
    ) as sock:
        target_sync_info = _ensure_remote_target_model(
            sock=sock,
            base_model_path=base_model_path,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            target_quantization=target_quantization,
            device_map=device_map,
            debug=False,
        )
        resolved_target_quantization = _normalize_quantization_mode(
            target_sync_info.get("selected_quantization", target_quantization), default=str(target_quantization or "8bit")
        )
        load_profile_data(
            runner=runner,
            tokenizer=tokenizer,
            max_depth=max_depth,
            draft_model_path=draft_model_path,
            base_model_path=base_model_path,
            bench_name=bench_name,
            device_name=device_name,
            server_name=server_name,
            target_quantization=resolved_target_quantization,
            draft_quantization=draft_quantization,
            question_file=question_file,
            fixed_depth=fixed_depth,
            debug=False,
            target_sock=sock,
            auto_target_profile=bool(auto_target_profile),
            draft_profile_force_refresh=bool(draft_profile_force_refresh)
            and not bool(chat_draft_profile_force_refresh_consumed),
            target_profile_force_refresh=bool(target_profile_force_refresh),
            target_profile_model_calls_per_count=int(target_profile_model_calls_per_count),
            target_profile_width_list=target_profile_width_list,
            target_profile_depth_list=target_profile_depth_list,
            target_profile_node_list=target_profile_node_list,
            draft_device_map=device_map,
            online_profile_update=bool(online_profile_update),
            online_profile_lr=float(online_profile_lr),
        )
        _emit({
            "type": "ready",
            "objective_selection_mode": str(objective_selection_mode),
            "constraint_target": str(constraint_target),
            "feasible_constraint_range_per_1m": feasible_constraint_range_per_1m,
            "feasible_tps_range": feasible_tps_range,
            "reference_tradeoff_curve_cs0_1": reference_tradeoff_curve_cs0_1,
            "reference_tradeoff_curve_by_constraint": reference_tradeoff_curve_by_constraint,
            "reference_cs_anchor_curve": reference_cs_anchor_curve,
            "reference_constraint_anchor_curve": reference_constraint_anchor_curve,
        })
        active_proactive_drafting = False
        active_chat_max_new_tokens = int(chat_max_new_tokens)
        active_constraint_target = str(constraint_target)
        active_metric_constraint_per_1m_token = (
            float(metric_constraint_per_1m_token) if metric_constraint_per_1m_token is not None else None
        )
        active_min_tps_constraint = (
            float(min_tps_constraint) if min_tps_constraint is not None and float(min_tps_constraint) > 0 else None
        )

        def _emit_status():
            _emit({
                "type": "status",
                "reference_tps": float(runner.reference_tps),
                "cost_sensitivity": float(runner.cost_sensitivity),
                "proactive_drafting": bool(active_proactive_drafting),
                "max_new_tokens": int(active_chat_max_new_tokens),
                "constraint_target": str(active_constraint_target),
                "metric_constraint_per_1m_token": (
                    float(active_metric_constraint_per_1m_token)
                    if active_metric_constraint_per_1m_token is not None
                    else None
                ),
                "min_tps_constraint": (
                    float(active_min_tps_constraint)
                    if active_min_tps_constraint is not None
                    else 0.0
                ),
                "feasible_constraint_range_per_1m": feasible_constraint_range_per_1m,
                "feasible_tps_range": feasible_tps_range,
                "reference_tradeoff_curve_cs0_1": reference_tradeoff_curve_cs0_1,
                "reference_tradeoff_curve_by_constraint": reference_tradeoff_curve_by_constraint,
                "reference_cs_anchor_curve": reference_cs_anchor_curve,
                "reference_constraint_anchor_curve": reference_constraint_anchor_curve,
                "last_line": "ok",
            })

        def _apply_set(payload_set: dict, emit_ack: bool = True):
            nonlocal active_proactive_drafting, active_chat_max_new_tokens, active_constraint_target, active_metric_constraint_per_1m_token, active_min_tps_constraint
            if "cost_sensitivity" in payload_set:
                try:
                    runner.cost_sensitivity = float(payload_set.get("cost_sensitivity"))
                except Exception:
                    pass
            if "proactive_drafting" in payload_set:
                active_proactive_drafting = bool(payload_set.get("proactive_drafting"))
            if "max_new_tokens" in payload_set:
                try:
                    active_chat_max_new_tokens = max(1, int(payload_set.get("max_new_tokens")))
                except Exception:
                    pass
            if "constraint_target" in payload_set:
                incoming_target = str(payload_set.get("constraint_target", active_constraint_target)).lower()
                if incoming_target in {"metric", "tps"}:
                    active_constraint_target = incoming_target
                    runner.constraint_target = incoming_target
            if "metric_constraint_per_1m_token" in payload_set:
                try:
                    active_metric_constraint_per_1m_token = float(payload_set.get("metric_constraint_per_1m_token"))
                    runner.metric_constraint_per_token = (
                        float(active_metric_constraint_per_1m_token) / 1_000_000.0
                        if active_constraint_target == "metric"
                        else None
                    )
                except Exception:
                    pass
            if "min_tps_constraint" in payload_set:
                try:
                    incoming_min_tps = float(payload_set.get("min_tps_constraint"))
                    active_min_tps_constraint = incoming_min_tps if incoming_min_tps > 0 else None
                    runner.min_tps_constraint = active_min_tps_constraint if active_constraint_target == "tps" else None
                except Exception:
                    pass
            if emit_ack:
                _emit({
                    "type": "settings_ack",
                    "cost_sensitivity": float(runner.cost_sensitivity),
                    "proactive_drafting": bool(active_proactive_drafting),
                    "max_new_tokens": int(active_chat_max_new_tokens),
                    "constraint_target": str(active_constraint_target),
                    "metric_constraint_per_1m_token": (
                        float(active_metric_constraint_per_1m_token)
                        if active_metric_constraint_per_1m_token is not None
                        else None
                    ),
                    "min_tps_constraint": (
                        float(active_min_tps_constraint)
                        if active_min_tps_constraint is not None
                        else 0.0
                    ),
                    "feasible_constraint_range_per_1m": feasible_constraint_range_per_1m,
                    "feasible_tps_range": feasible_tps_range,
                    "reference_tradeoff_curve_cs0_1": reference_tradeoff_curve_cs0_1,
                    "reference_tradeoff_curve_by_constraint": reference_tradeoff_curve_by_constraint,
                    "reference_cs_anchor_curve": reference_cs_anchor_curve,
                    "reference_constraint_anchor_curve": reference_constraint_anchor_curve,
                })

        def _poll_live_controls() -> bool:
            """
            Non-blocking stdin poll while a chat turn is running.
            Returns True when stop/quit/exit is requested.
            """
            while True:
                try:
                    ready, _, _ = select.select([sys.stdin], [], [], 0.0)
                except Exception:
                    return False
                if not ready:
                    return False
                live_line = sys.stdin.readline()
                if not live_line:
                    return False
                live_line = live_line.strip()
                if not live_line:
                    continue
                try:
                    live_payload = json.loads(live_line)
                except Exception:
                    live_payload = {"cmd": "chat", "text": live_line}
                live_cmd = str(live_payload.get("cmd", "chat")).lower()
                if live_cmd in {"stop", "quit", "exit"}:
                    return True
                if live_cmd == "set":
                    _apply_set(live_payload, emit_ack=False)
                elif live_cmd == "status":
                    _emit_status()
                # Ignore new chat commands while the current turn is running.

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                payload = {"cmd": "chat", "text": line}
            cmd = str(payload.get("cmd", "chat")).lower()
            if cmd in {"stop", "quit", "exit"}:
                _emit({"type": "stopped"})
                break
            if cmd == "status":
                _emit_status()
                continue
            if cmd == "set":
                _apply_set(payload, emit_ack=True)
                continue
            if cmd != "chat":
                _emit({"type": "error", "message": f"Unsupported cmd: {cmd}"})
                continue

            user_text = str(payload.get("text", "")).strip()
            if "cost_sensitivity" in payload:
                try:
                    runner.cost_sensitivity = float(payload.get("cost_sensitivity"))
                except Exception:
                    pass
            if "proactive_drafting" in payload:
                active_proactive_drafting = bool(payload.get("proactive_drafting"))
            if "max_new_tokens" in payload:
                try:
                    active_chat_max_new_tokens = max(1, int(payload.get("max_new_tokens")))
                except Exception:
                    pass
            if "constraint_target" in payload:
                incoming_target = str(payload.get("constraint_target", active_constraint_target)).lower()
                if incoming_target in {"metric", "tps"}:
                    active_constraint_target = incoming_target
                    runner.constraint_target = incoming_target
            if "metric_constraint_per_1m_token" in payload:
                try:
                    active_metric_constraint_per_1m_token = float(payload.get("metric_constraint_per_1m_token"))
                    runner.metric_constraint_per_token = (
                        float(active_metric_constraint_per_1m_token) / 1_000_000.0
                        if active_constraint_target == "metric"
                        else None
                    )
                except Exception:
                    pass
            if "min_tps_constraint" in payload:
                try:
                    incoming_min_tps = float(payload.get("min_tps_constraint"))
                    active_min_tps_constraint = incoming_min_tps if incoming_min_tps > 0 else None
                    runner.min_tps_constraint = active_min_tps_constraint if active_constraint_target == "tps" else None
                except Exception:
                    pass
            if not user_text:
                _emit({"type": "chat_reply", "reply": ""})
                continue

            try:
                query_profile_update_overhead_sec = 0.0
                target_verify_per_nnodes_query = {}
                runner.reset_timing_stats()
                runner.reset_kv()
                conv.append_message(conv.roles[0], user_text)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt() + " "
                input_ids = tokenizer([prompt]).input_ids
                input_ids_t = torch.as_tensor(input_ids).to("cuda")

                send_json_with_size(sock, {"type": "init", "input_ids": input_ids[0]})
                reply, _ = recv_json_with_size(sock)
                if reply.get("type") != "init_ok":
                    _emit({"type": "error", "message": f"init failed: {reply}"})
                    continue
                current_next_token = reply["next_token"]
                output_tokens: List[int] = []
                output_trace: List[dict] = []
                new_token_count = 0
                special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
                use_proactive_tree = False
                proactive_tree = None
                trace_token_buffer: List[int] = []
                trace_decoded_buffer = ""
                step_idx = 0
                cum_new_tokens = 0
                cum_wall_time_sec = 0.0
                cum_tree_build_sec = 0.0
                cum_target_verify_sec = 0.0
                cum_d2t_bytes = 0.0
                cum_t2d_bytes = 0.0
                cum_draft_energy_kwh = 0.0
                # UI/text rendering overlap guard:
                # previous step's next_token can reappear as next step's accepted head.
                last_step_next_token_for_overlap = None

                def _append_token_trace(token_id: int, origin: str):
                    nonlocal trace_token_buffer, trace_decoded_buffer
                    try:
                        tid = int(token_id)
                    except Exception:
                        return
                    if tid in special_ids:
                        return
                    # decode ,
                    # decode / .
                    prev_decoded = trace_decoded_buffer
                    trace_token_buffer.append(tid)
                    try:
                        new_decoded = tokenizer.decode(
                            trace_token_buffer,
                            clean_up_tokenization_spaces=False,
                            spaces_between_special_tokens=False,
                        )
                    except Exception:
                        new_decoded = prev_decoded
                    if new_decoded is None:
                        new_decoded = prev_decoded
                    if new_decoded.startswith(prev_decoded):
                        piece = new_decoded[len(prev_decoded):]
                    else:
                        # decode fallback
                        try:
                            piece = tokenizer.decode(
                                [tid],
                                clean_up_tokenization_spaces=False,
                                spaces_between_special_tokens=False,
                            )
                        except Exception:
                            piece = ""
                    trace_decoded_buffer = str(new_decoded)
                    output_tokens.append(tid)
                    output_trace.append({"text": str(piece), "origin": str(origin)})

                while True:
                    if _poll_live_controls():
                        _emit({"type": "stopped"})
                        return
                    if new_token_count >= int(active_chat_max_new_tokens):
                        break
                    step_idx += 1
                    step_wall_start = time.time()
                    if runner.gpu_monitor:
                        runner.gpu_monitor.start_monitoring()
                    tree_build_start = time.time()
                    current_tree_is_proactive = False
                    if use_proactive_tree and proactive_tree is not None:
                        draft_ids = proactive_tree["draft_ids"]
                        draft_pos = proactive_tree["draft_pos"]
                        tree_mask = proactive_tree["tree_mask"]
                        parent = proactive_tree["parent"]
                        tree_depth = proactive_tree["tree_depth"]
                        final_nnodes = proactive_tree["final_nnodes"]
                        depth_widths = proactive_tree["depth_widths"]
                        node_meta = proactive_tree.get("node_meta")
                        runner.last_sum_expected_accepted_length = proactive_tree.get("expected_accept_length")
                        runner.last_accept_length_scale_used = float(
                            proactive_tree.get("accept_length_scale_used", 1.0)
                        )
                        use_proactive_tree = False
                        proactive_tree = None
                        current_tree_is_proactive = True
                    else:
                        tree_build_start = time.time()
                        draft_ids, draft_pos, tree_mask, parent, tree_depth, final_nnodes, depth_widths, node_meta = build_tree_with_next_token(
                            runner,
                            input_ids_t,
                            nodes,
                            max_depth,
                            current_next_token,
                            tokenizer,
                            False,
                            False,
                            per_token_probability_bound=per_token_probability_bound,
                            per_path_probability_bound=per_path_probability_bound,
                            min_width=min_width,
                            fixed_width=fixed_width,
                            fixed_width_value=fixed_width_value,
                            fixed_nnodes=fixed_nnodes,
                            fixed_depth=fixed_depth,
                        )
                    tree_build_end = time.time()
                    tree_build_time_sec = max(0.0, float(tree_build_end - tree_build_start))
                    send_start_time = time.time()
                    d2t_bytes = send_json_with_size(
                        sock,
                        {
                            "type": "tree_step",
                            "draft_input_ids": draft_ids[0].tolist(),
                            "draft_position_ids": draft_pos.tolist(),
                            "tree_attention_mask": tree_mask.tolist(),
                            "parent": parent.tolist(),
                        },
                    )
                    # proactive drafting :
                    pending_proactive = None
                    proactive_thread = None
                    proactive_stop_flag = None
                    proactive_result = {
                        "tree": None,
                        "error": None,
                        "elapsed_sec": None,
                        "head_token": None,
                        "expected_path_tokens": None,
                        "path_match": None,
                    }
                    if bool(active_proactive_drafting):
                        try:
                            proactive_stop_flag = threading.Event()
                            node_tokens = draft_ids[0].tolist()[1:]
                            parent_list = parent.tolist() if isinstance(parent, torch.Tensor) else list(parent)
                            proactive_path, proactive_path_prob = _select_proactive_path(node_tokens, parent_list, node_meta)
                            if proactive_path:
                                proactive_path = [int(current_next_token)] + [int(tok) for tok in proactive_path]
                                proactive_result["expected_path_tokens"] = list(proactive_path)
                                def _build_proactive_tree():
                                    try:
                                        proactive_result["tree"] = build_proactive_tree_from_path(
                                            runner=runner,
                                            base_input_ids=input_ids_t,
                                            path_tokens=proactive_path,
                                            nodes=nodes,
                                            max_depth=max_depth,
                                            tokenizer=tokenizer,
                                            debug=False,
                                            print_tree=False,
                                            per_token_probability_bound=per_token_probability_bound,
                                            per_path_probability_bound=per_path_probability_bound,
                                            min_width=min_width,
                                            fixed_width=fixed_width,
                                            fixed_width_value=fixed_width_value,
                                            fixed_nnodes=fixed_nnodes,
                                            fixed_depth=fixed_depth,
                                            stop_flag=proactive_stop_flag,
                                            head_token_holder=proactive_result,
                                        )
                                    except Exception as e:
                                        proactive_result["error"] = e
                                proactive_thread = threading.Thread(target=_build_proactive_tree, daemon=True)
                                proactive_thread.start()
                        except Exception:
                            proactive_thread = None
                    reply, t2d_bytes = recv_json_with_size(sock)
                    gpu_stats = None
                    if runner.gpu_monitor:
                        runner.gpu_monitor.stop_monitoring()
                        gpu_stats = runner.gpu_monitor.get_stats()
                    recv_end_time = time.time()
                    if reply.get("type") != "verify_result":
                        _emit({"type": "error", "message": f"verify failed: {reply}"})
                        break
                    # Apply live set updates before deciding next-step proactive usage.
                    if _poll_live_controls():
                        _emit({"type": "stopped"})
                        return
                    accepted_tokens: List[int] = reply["accepted_tokens"]
                    accept_length: int = reply["accept_length"]
                    best_ids = reply.get("best_ids", [])
                    next_token: int = reply["next_token"]
                    accepted_plus_next_tokens = reply.get("accepted_plus_next_tokens", None)
                    eos_reached: bool = reply["eos_reached"]
                    target_energy_rate_per_sec = reply.get("target_energy_rate_per_sec", None)
                    # proactive ( step )
                    if bool(active_proactive_drafting):
                        canceled = False
                        if proactive_thread is not None:
                            cancel_reason = None
                            if accept_length != tree_depth:
                                cancel_reason = "accept_length_mismatch"
                            else:
                                proactive_head = proactive_result.get("head_token")
                                if proactive_head is None or next_token != proactive_head:
                                    cancel_reason = "head_mismatch"
                                else:
                                    expected_path_tokens = proactive_result.get("expected_path_tokens") or []
                                    if expected_path_tokens:
                                        try:
                                            accepted_path_tokens = [int(tok) for tok in accepted_tokens]
                                            expected_path_tokens = [int(tok) for tok in expected_path_tokens]
                                            proactive_result["path_match"] = (
                                                accepted_path_tokens == expected_path_tokens
                                            )
                                        except Exception:
                                            proactive_result["path_match"] = False
                                    if proactive_result.get("path_match") is not True:
                                        cancel_reason = "path_mismatch"
                            if cancel_reason is not None:
                                canceled = True
                                proactive_stop_flag.set()
                                runner.reset_proactive_kv()
                                pending_proactive = None
                            else:
                                proactive_thread.join()
                                pending_proactive = proactive_result["tree"]
                        if (
                            pending_proactive
                            and accept_length == tree_depth
                            and next_token == pending_proactive.get("head_token")
                            and proactive_result.get("path_match") is True
                        ):
                            use_proactive_tree = True
                            proactive_tree = pending_proactive
                            runner.draft_stable_kv = runner.proactive_kv
                        else:
                            if not canceled and proactive_thread is not None:
                                proactive_stop_flag.set()
                            runner.reset_proactive_kv()
                            use_proactive_tree = False
                            proactive_tree = None
                    else:
                        runner.reset_proactive_kv()
                        use_proactive_tree = False
                        proactive_tree = None

                    target_recv_end_time = reply.get("target_recv_end_time", None)
                    target_send_start_time = reply.get("target_send_start_time", None)
                    draft_to_target_time = None
                    target_to_draft_time = None
                    if target_recv_end_time is not None:
                        draft_to_target_time = max(0.05, (target_recv_end_time - send_start_time) * 1000.0)
                    if target_send_start_time is not None:
                        target_to_draft_time = max(0.05, (recv_end_time - target_send_start_time) * 1000.0)

                    total_time = max(0.0, (target_recv_end_time - send_start_time)) if target_recv_end_time is not None else 0.0
                    transfer_time = (
                        (draft_to_target_time + target_to_draft_time) / 1000.0
                        if draft_to_target_time is not None and target_to_draft_time is not None
                        else None
                    )
                    runner.prev_tree_final_nnodes = final_nnodes
                    runner.prev_tree_depth = tree_depth
                    runner.prev_tree_total_target_time = total_time
                    runner.prev_tree_transfer_time = transfer_time
                    runner.prev_tree_accept_length = accept_length
                    if draft_to_target_time is not None and final_nnodes > 0:
                        runner.per_token_draft_to_target_transfer_time = (draft_to_target_time / 1000.0) / final_nnodes
                        runner.per_token_draft_to_target_bytes = float(d2t_bytes) / final_nnodes
                    if target_to_draft_time is not None and accept_length > 0:
                        runner.per_token_target_to_draft_transfer_time = (target_to_draft_time / 1000.0) / accept_length
                        runner.per_token_target_to_draft_bytes = float(t2d_bytes) / accept_length

                    input_ids_t = torch.cat([
                        input_ids_t,
                        torch.tensor([accepted_tokens], device=input_ids_t.device, dtype=torch.long),
                    ], dim=-1)
                    accept_origin = "proactive_accept" if current_tree_is_proactive else "draft_accept"
                    accepted_tokens_for_emit = list(accepted_tokens or [])
                    next_token_for_emit = int(next_token)
                    if isinstance(accepted_plus_next_tokens, list) and accepted_plus_next_tokens:
                        try:
                            parsed_combo = [int(tok) for tok in accepted_plus_next_tokens]
                            next_token_for_emit = int(parsed_combo[-1])
                            if len(parsed_combo) >= 2:
                                accepted_tokens_for_emit = parsed_combo[:-1]
                        except Exception:
                            pass
                    if (
                        last_step_next_token_for_overlap is not None
                        and accepted_tokens_for_emit
                        and int(accepted_tokens_for_emit[0]) == int(last_step_next_token_for_overlap)
                    ):
                        # Remove only the overlapped head that was already emitted
                        # as previous step's server_new token.
                        accepted_tokens_for_emit = accepted_tokens_for_emit[1:]
                    for tok in accepted_tokens_for_emit:
                        _append_token_trace(tok, accept_origin)
                    # Avoid duplicated rendering when next_token overlaps emitted accepted tail.
                    if (
                        not accepted_tokens_for_emit
                        or int(next_token_for_emit) != int(accepted_tokens_for_emit[-1])
                    ):
                        _append_token_trace(next_token_for_emit, "server_new")
                    last_step_next_token_for_overlap = int(next_token_for_emit)
                    new_token_count += accept_length + 1
                    current_next_token = next_token

                    step_new_tokens = int(max(0, accept_length + 1))
                    cum_new_tokens += step_new_tokens
                    cum_wall_time_sec += max(0.0, recv_end_time - step_wall_start)
                    cum_tree_build_sec += max(0.0, tree_build_time_sec)
                    target_verification_sec = 0.0
                    try:
                        if reply.get("target_verification_time_ms", None) is not None:
                            target_verification_sec = max(
                                0.0, float(reply.get("target_verification_time_ms", 0.0) or 0.0) / 1000.0
                            )
                        elif reply.get("target_verification_time", None) is not None:
                            # backward compatibility: some servers may send ms in this key
                            target_verification_sec = max(
                                0.0, float(reply.get("target_verification_time", 0.0) or 0.0) / 1000.0
                            )
                        elif reply.get("target_verification_time_sec", None) is not None:
                            target_verification_sec = max(
                                0.0, float(reply.get("target_verification_time_sec", 0.0) or 0.0)
                            )
                        elif (
                            reply.get("target_send_start_time", None) is not None
                            and reply.get("target_recv_end_time", None) is not None
                        ):
                            # fallback: if only target-side wall timestamps are available
                            target_verification_sec = max(
                                0.0,
                                float(reply.get("target_recv_end_time", 0.0) or 0.0)
                                - float(reply.get("target_send_start_time", 0.0) or 0.0),
                            )
                    except Exception:
                        target_verification_sec = 0.0
                    cum_target_verify_sec += target_verification_sec
                    if target_verification_sec > 0:
                        ms_val = float(target_verification_sec) * 1000.0
                        tvals = target_verify_per_nnodes_query.setdefault(int(final_nnodes), [])
                        tvals.append(ms_val)
                    cum_d2t_bytes += max(0.0, float(d2t_bytes))
                    cum_t2d_bytes += max(0.0, float(t2d_bytes))
                    total_gpu_power_w = _extract_total_gpu_power_w(gpu_stats)
                    if total_gpu_power_w > 0:
                        # Energy(J) = W * sec, kWh = J / 3,600,000
                        step_draft_energy_kwh = (total_gpu_power_w * max(0.0, float(tree_build_time_sec))) / 3_600_000.0
                        cum_draft_energy_kwh += max(0.0, float(step_draft_energy_kwh))
                    runner.update_target_objective_rate(target_energy_rate_per_sec)

                    throughput_tps = (
                        float(cum_new_tokens) / max(1e-9, float(cum_wall_time_sec))
                        if cum_new_tokens > 0
                        else 0.0
                    )
                    if no_draft_cost:
                        draft_cost_total = 0.0
                    elif str(objective_metric).lower() in {"total_cost", "api_cost"}:
                        if bool(getattr(runner, "bill_draft_as_target_gpu", False)):
                            draft_cost_total = float(cum_tree_build_sec) * float(target_per_sec_cost)
                        else:
                            draft_cost_total = float(cum_draft_energy_kwh) * float(draft_electricity_cost_per_kwh)
                    else:
                        draft_cost_total = float(cum_tree_build_sec) * float(draft_per_sec_cost)
                    target_cost_total = float(cum_target_verify_sec) * float(target_per_sec_cost)
                    inbound_comm_cost_total = (
                        (float(cum_d2t_bytes) / float(1024 ** 3)) * float(user_communication_cost_per_gb)
                    )
                    user_outbound_comm_cost_total = (
                        (float(cum_t2d_bytes) / float(1024 ** 3)) * float(user_communication_cost_per_gb)
                    )
                    outbound_comm_cost_total = (
                        (float(cum_t2d_bytes) / float(1024 ** 3)) * float(cloud_outbound_cost_per_gb)
                    )
                    comm_cost_total = inbound_comm_cost_total + user_outbound_comm_cost_total + outbound_comm_cost_total
                    draft_cost_per_1m = (
                        (draft_cost_total / float(cum_new_tokens)) * 1_000_000.0 if cum_new_tokens > 0 else 0.0
                    )
                    target_cost_per_1m = (
                        (target_cost_total / float(cum_new_tokens)) * 1_000_000.0 if cum_new_tokens > 0 else 0.0
                    )
                    comm_cost_per_1m = (
                        (comm_cost_total / float(cum_new_tokens)) * 1_000_000.0 if cum_new_tokens > 0 else 0.0
                    )
                    inbound_comm_cost_per_1m = (
                        (inbound_comm_cost_total / float(cum_new_tokens)) * 1_000_000.0 if cum_new_tokens > 0 else 0.0
                    )
                    outbound_comm_cost_per_1m = (
                        (outbound_comm_cost_total / float(cum_new_tokens)) * 1_000_000.0 if cum_new_tokens > 0 else 0.0
                    )
                    user_outbound_comm_cost_per_1m = (
                        (user_outbound_comm_cost_total / float(cum_new_tokens)) * 1_000_000.0 if cum_new_tokens > 0 else 0.0
                    )
                    draft_energy_per_1m = (
                        (float(cum_draft_energy_kwh) / float(cum_new_tokens)) * 1_000_000.0
                        if cum_new_tokens > 0
                        else 0.0
                    )

                    stop_marker_reached = False
                    if conv.stop_str:
                        try:
                            raw_partial = tokenizer.decode(
                                output_tokens,
                                spaces_between_special_tokens=False,
                            )
                            stop_marker_reached = raw_partial.find(conv.stop_str) > 0
                        except Exception:
                            stop_marker_reached = False
                    partial_decoded = _decode_output(output_tokens)
                    parent_list = parent.tolist() if isinstance(parent, torch.Tensor) else list(parent)
                    accepted_node_ids = []
                    if isinstance(best_ids, list):
                        # best_ids is indexed on target-side draft_input_ids where index 0 is the synthetic head token.
                        # UI tree_parent indices are on the original parent array domain (head excluded),
                        # so map accepted ids by dropping head and shifting by -1.
                        for idx, nid in enumerate(best_ids):
                            try:
                                nid_i = int(nid)
                            except Exception:
                                continue
                            if idx == 0:
                                # synthetic head node (rendered as root=-1 on UI)
                                continue
                            mapped = nid_i - 1
                            if mapped >= 0:
                                accepted_node_ids.append(mapped)
                    _emit({
                        "type": "chat_partial",
                        "reply": partial_decoded,
                        "token_trace": output_trace,
                        "stats": {
                            "step": int(step_idx),
                            "new_tokens": int(cum_new_tokens),
                            "throughput": float(throughput_tps),
                            "gpu_energy": float(draft_energy_per_1m),
                            "draft_cost": float(draft_cost_per_1m),
                            "target_cost": float(target_cost_per_1m),
                            "user_communication_inbound_cost": float(inbound_comm_cost_per_1m),
                            "cloud_outbound_cost": float(outbound_comm_cost_per_1m),
                            "user_communication_outbound_cost": float(user_outbound_comm_cost_per_1m),
                            "communication_cost": float(comm_cost_per_1m),
                            "tree_depth": int(tree_depth),
                            "final_nnodes": int(final_nnodes),
                            "depth_widths": [int(w) for w in (depth_widths or [])],
                            "accept_length": int(accept_length),
                            "tree_parent": [int(p) for p in parent_list],
                            "accepted_node_ids": accepted_node_ids,
                            "tree_origin": ("proactive_accept" if current_tree_is_proactive else "draft_accept"),
                        },
                    })

                    if stop_marker_reached or eos_reached or new_token_count >= int(active_chat_max_new_tokens):
                        break

                decoded = _decode_output(output_tokens)
                conv.messages[-1][-1] = decoded
                observed_width_ms_query = {}
                for _w, _times in getattr(runner, "question_width_times", {}).items():
                    try:
                        w_key = int(_w)
                    except Exception:
                        continue
                    if not isinstance(_times, list) or (not _times):
                        continue
                    observed_width_ms_query[w_key] = []
                    for _sec in _times:
                        try:
                            ms_val = float(_sec) * 1000.0
                            if np.isfinite(ms_val) and ms_val >= 0:
                                observed_width_ms_query[w_key].append(ms_val)
                        except Exception:
                            continue
                online_profile_update_result = _apply_online_profile_updates_and_flush(
                    runner,
                    observed_width_ms=observed_width_ms_query,
                    observed_nnodes_ms=target_verify_per_nnodes_query,
                    debug=False,
                )
                query_profile_update_overhead_sec = float(
                    online_profile_update_result.get("overhead_sec", 0.0) or 0.0
                )
                wall_tps = (
                    float(cum_new_tokens) / max(1e-9, float(cum_wall_time_sec))
                    if cum_new_tokens > 0
                    else 0.0
                )
                final_stats = {
                    "new_tokens": int(cum_new_tokens),
                    "throughput": float(wall_tps),
                    "throughput_effective": float(wall_tps),
                    "throughput_wall": float(wall_tps),
                    "profile_update_overhead_seconds": float(query_profile_update_overhead_sec),
                    "draft_cost": float(draft_cost_per_1m) if "draft_cost_per_1m" in locals() else 0.0,
                    "target_cost": float(target_cost_per_1m) if "target_cost_per_1m" in locals() else 0.0,
                    "communication_cost": float(comm_cost_per_1m) if "comm_cost_per_1m" in locals() else 0.0,
                    "gpu_energy": float(draft_energy_per_1m) if "draft_energy_per_1m" in locals() else 0.0,
                }
                _emit({
                    "type": "chat_reply",
                    "reply": decoded,
                    "token_trace": output_trace,
                    "final_stats": final_stats,
                    "feasible_constraint_range_per_1m": feasible_constraint_range_per_1m,
                    "feasible_tps_range": feasible_tps_range,
                    "reference_tradeoff_curve_cs0_1": reference_tradeoff_curve_cs0_1,
                    "reference_tradeoff_curve_by_constraint": reference_tradeoff_curve_by_constraint,
                    "reference_cs_anchor_curve": reference_cs_anchor_curve,
                    "reference_constraint_anchor_curve": reference_constraint_anchor_curve,
                })
            except Exception as e:
                _emit({"type": "error", "message": str(e)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=26001)
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--draft-model-path", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str, default=None, help="Optional tokenizer model/path. Default: base-model-path")
    parser.add_argument(
        "--bench-name",
        type=str,
        choices=["mt_bench", "gsm8k", "humaneval", "ifeval", "math-500", "math500", "cnn_dailymail"],
        required=True,
    )
    parser.add_argument("--question-file", type=str, default=None)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--nodes", type=int, default=50)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--device-map", type=str, default="cuda:0")
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Force draft model 4-bit quantization.",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Force draft model 8-bit quantization.",
    )
    parser.add_argument(
        "--target-quantization",
        type=str,
        choices=["auto", "none", "4bit", "8bit"],
        default="auto",
        help=(
            "Target server reload quantization mode. "
            "auto: start from 8bit for 70B+/72B targets, otherwise start from none."
        ),
    )
    parser.add_argument("--answer-file", type=str, default=None, help="Output answer file")
    parser.add_argument("--model-id", type=str, default="llama-2-chat", help="Model ID")
    parser.add_argument("--num-choices", type=int, default=1, help="Number of completion choices")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug output (DRAFT-DEBUG messages)")
    parser.add_argument("--print-tree", action="store_true", default=False, help="Print tree structure during expansion")
    parser.add_argument("--per-token-probability-bound", type=float, default=0.0, help="Probability bound for each token (default: 0.0)")
    parser.add_argument("--per-path-probability-bound", type=float, default=0.0, help="Probability bound for each path (default: 0.0)")
    parser.add_argument("--device-name", type=str, default=None, help="Device name for profiling (e.g., rtx5080). If provided, profiles width timing after warmup.")
    parser.add_argument(
        "--draft-per-hour-cost",
        type=float,
        default=0.0,
        help=(
            "(Legacy) Draft model time-based $/hour. "
            "Ignored when --objective-metric=total_cost "
            "(total_cost uses --draft-electricity-cost-per-kwh + measured GPU power)."
        ),
    )
    parser.add_argument(
        "--draft-electricity-cost-per-kwh",
        type=float,
        default=0.2,
        help=(
            "Draft GPU electricity price in USD/kWh used by --objective-metric=total_cost "
            "(default: 0.2)."
        ),
    )
    parser.add_argument(
        "--target-per-hour-cost",
        type=float,
        default=1.208,
        help=(
            "Target server/model billing in $/hour (default: 1.208). "
            "Used by total_cost/api_cost as target time cost; ignored by energy_* objectives."
        ),
    )
    parser.add_argument(
        "--user-communication-cost-per-gb",
        type=float,
        default=0.09,
        help="User communication transfer price in $/GB (default: 0.09).",
    )
    parser.add_argument(
        "--cloud-outbound-cost-per-gb",
        type=float,
        default=0.09,
        help="Cloud outbound transfer price in $/GB (default: 0.09).",
    )
    parser.add_argument("--accept-length-margin", type=float, default=0.05, help="Conservative margin for expected accept length (default: 0.05)")
    parser.add_argument(
        "--objective-selection-mode",
        type=str,
        choices=["blend", "constraint"],
        default="blend",
        help="Tree selection mode: blend (current) or constraint (maximize TPS under metric constraint).",
    )
    parser.add_argument(
        "--metric-constraint-per-1m-token",
        type=float,
        default=None,
        help="Constraint mode target metric per 1M tokens. If omitted, auto-inferred from blend(cs=0.5) reference.",
    )
    parser.add_argument(
        "--total-metric-cap",
        type=float,
        default=float("inf"),
        help="Stop decoding when total metric budget is exhausted ($ for cost, kWh for energy). Default: inf",
    )
    parser.add_argument(
        "--reference-test-mode",
        action="store_true",
        default=False,
        help="Run reference range validation probes and exit.",
    )
    parser.add_argument(
        "--reference-test-runs",
        type=int,
        default=5,
        help="Number of probe runs for reference range validation (default: 5).",
    )
    parser.add_argument(
        "--reference-test-output-json",
        type=str,
        default=None,
        help="Optional output path for reference range validation report JSON.",
    )
    parser.add_argument(
        "--reference-cs-curve-rounds",
        type=int,
        default=20,
        help="Reference full-query rounds used to build per-step cs anchor curve (default: 20).",
    )
    parser.add_argument(
        "--reference-max-steps-limit",
        type=int,
        default=1,
        help="Max step rounds per query during reference cache generation (default: 1).",
    )
    parser.add_argument(
        "--reference-constraint-multipliers",
        type=str,
        default="0.8,1.0,1.2",
        help="Constraint-mode reference sweep multipliers (comma-separated). "
             "Anchors are auto-inferred center metric * multipliers.",
    )
    parser.add_argument("--auto-profile", action="store_true", default=True,
                        help="Draft/Target profile file is missing, auto-generate (default: on)")
    parser.add_argument("--disable-auto-profile", action="store_true", default=False,
                        help="Draft/Target profile Disable auto-generation")
    parser.add_argument("--draft-profile-force-refresh", action="store_true", default=False,
                        help="Draft profile Force regeneration even if the file exists")
    parser.add_argument(
        "--force-profile-refresh",
        action="store_true",
        default=False,
        help="Draft/Target profile Re-measure and replace both files even if they already exist",
    )
    parser.add_argument(
        "--draft-profile-model-calls-per-count",
        type=int,
        default=100,
        help="Draft profile target number of model.model() calls per width during generation (default: 100)",
    )
    parser.add_argument(
        "--draft-profile-width-list",
        type=str,
        default="10,20,30,40,50,60,70,80,90,100,110,120,130,140,150",
        help="Draft profile width list (default: 10~150, 10-step increments, comma-separated)",
    )
    parser.add_argument(
        "--draft-profile-only",
        action="store_true",
        default=False,
        help="Skip target connection/generation and create only the draft profile during profiling",
    )
    parser.add_argument("--target-profile-force-refresh", action="store_true", default=False,
                        help="Target profile Force regeneration even if the file exists")
    parser.add_argument("--target-profile-model-calls-per-count", type=int, default=10,
                        help="during automatic target profiling (fixed width/depth and nnodes) model call repeats per combination (default: 10)")
    parser.add_argument("--target-profile-node-list", type=str, default="10,20,30,40,50,60,70,80,90,100,110,120,130,140,150",
                        help="Automatic target profile node list (default: 10~150, 10-step increments, comma-separated)")
    parser.add_argument(
        "--profile-only",
        action="store_true",
        default=False,
        help="Create/load draft/target profiles only (data/profile) and exit without reference/answer generation.",
    )
    parser.add_argument(
        "--profile-only-report-json",
        type=str,
        default=None,
        help="Optional JSON output path for --profile-only summary.",
    )
    parser.add_argument(
        "--reference-force-refresh",
        action="store_true",
        default=False,
        help="Ignore existing reference cache and rebuild reference only.",
    )
    parser.add_argument(
        "--reference-only-exit-after-cache",
        action="store_true",
        default=False,
        help="Build/load reference cache and exit without decoding benchmark questions.",
    )
    parser.add_argument(
        "--disable-online-profile-update",
        action="store_true",
        default=False,
        help="Disable online profile table updates at each query end.",
    )
    parser.add_argument(
        "--disable-accept-length-calibration",
        action="store_true",
        default=False,
        help="Disable online accept-length ratio calibration during decoding.",
    )
    parser.add_argument(
        "--disable-target-time-calibration",
        action="store_true",
        default=False,
        help="Disable online target-verification time ratio calibration during decoding.",
    )
    parser.add_argument(
        "--disable-online-calibration",
        action="store_true",
        default=False,
        help="Disable online accept-length and target-time calibration during decoding.",
    )
    parser.add_argument(
        "--online-profile-lr",
        type=float,
        default=ONLINE_PROFILE_LR_DEFAULT,
        help="Learning rate for online profile updates (old + lr * delta). Default: 0.05",
    )
    parser.add_argument(
        "--chat-mode",
        action="store_true",
        default=False,
        help="Run persistent stdin/stdout chat mode using hybrid speculative decoding.",
    )
    parser.add_argument(
        "--chat-max-new-tokens",
        type=int,
        default=512,
        help="Max generated tokens per chat turn in --chat-mode (default: 512).",
    )
    def parse_cost_sensitivity(value):
        """Parse cost sensitivity in [0, 1]."""
        try:
            v = float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid cost sensitivity value: {value}. Use a number in [0, 1].")
        if v < 0.0 or v > 1.0:
            raise argparse.ArgumentTypeError(f"Invalid cost sensitivity value: {value}. Must be in [0, 1].")
        return v
    
    parser.add_argument("--cost-sensitivity", type=parse_cost_sensitivity, default=0.0, help="Cost sensitivity weight in [0, 1] (0: TPS-focused, 1: objective-cost-focused).")
    parser.add_argument("--min-width", type=int, default=1, help="Minimum width for next_width selection (default: 1)")
    parser.add_argument("--fixed-depth", action="store_true", default=False, help="If set, expand depth until max_depth ignoring latency/cost condition")
    parser.add_argument("--fixed-nnodes", action="store_true", default=False, help="If set, fix final_nnodes as nodes")
    parser.add_argument("--fixed-width", action="store_true", default=False, help="If set, always use max_nnodes as current_width")
    parser.add_argument("--fixed-width-value", type=int, default=None, help="Fixed width value (overrides --fixed-width if set)")
    parser.add_argument("--opt-tree", action="store_true", default=False, help="Use weight-sum-based expansion (opt-tree)")
    parser.add_argument("--server-name", type=str, default="rtxproa6000", help="Server name used for target profile file naming (default: rtxproa6000)")
    parser.add_argument("--enable-gpu-monitor", action="store_true", default=False, help="Enable GPU monitoring")
    parser.add_argument("--gpu-monitor-interval", type=float, default=0.05, help="GPU monitoring interval (s) (default: 0.05)")
    parser.add_argument("--fix-gpu-clock", action="store_true", default=False, help="Fix GPU clock with nvidia-smi application clocks")
    parser.add_argument("--gpu-graphics-clock-mhz", type=int, default=None, help="Fixed graphics clock (MHz). Use with --fix-gpu-clock")
    parser.add_argument("--gpu-memory-clock-mhz", type=int, default=None, help="Fixed memory clock (MHz). Use with --fix-gpu-clock")
    parser.add_argument("--enable-cpu-monitor", action="store_true", default=False, help="Enable CPU power monitoring")
    parser.add_argument("--proactive-drafting", action="store_true", default=False, help="Enable proactive drafting (default: False)")
    parser.add_argument("--proactive-threshold", type=float, default=0.0, help="Path prob threshold to start proactive drafting (default: 0.0)")
    parser.add_argument("--adaptive-proactive-threshold", action="store_true", default=False, help="Enable adaptive proactive drafting decision based on expected gain/loss")
    parser.add_argument(
        "--disable-proactive-budget",
        action="store_true",
        default=False,
        help="Disable proactive pre-reply budget/pause control while keeping proactive start gating enabled.",
    )
    parser.add_argument(
        "--join-canceled-proactive-before-tree-build",
        action="store_true",
        default=False,
        help="Wait for canceled proactive work to fully exit before starting the next main draft tree build.",
    )
    parser.add_argument("--no-draft-cost", action="store_true", default=False, help="Ignore draft tree build cost in objective (latency still used)")
    parser.add_argument("--min-tps-constraint", type=float, default=0.0, help="Minimum predicted throughput (tok/s) required in constraint mode; 0 disables.")
    parser.add_argument("--constraint-target", type=str, default="metric", choices=["metric", "tps"], help="Constraint target in constraint mode: metric budget or minimum TPS.")
    parser.add_argument(
        "--objective-metric",
        type=str,
        choices=["cost", "total_cost", "api_cost", "draft_energy", "target_energy"],
        default="total_cost",
        help=(
            "Objective metric for tree/proactive decisions. "
            "total_cost= draft electricity + target time cost + communication; "
            "api_cost= target time cost + outbound communication; "
            "draft_energy/target_energy= energy (kWh) objectives."
        ),
    )
    parser.add_argument("--server-only-baseline-json", type=str, default=None, help="Server-only SD baseline metrics JSON path")
    parser.add_argument("--disable-server-only", action="store_true", default=False, help="Disable automatic server-only mode switching")
    parser.add_argument("--force-server-only", action="store_true", default=False, help="Force server-only speculative decoding mode")
    parser.add_argument("--force-server-only-ar", action="store_true", default=False, help="Force target-side autoregressive model.forward server-only mode")
    parser.add_argument("--server-only-ar-turn-rpc", dest="server_only_ar_turn_rpc", action="store_true", default=True, help="For target-side server-only AR, return one full turn per RPC. This is the default.")
    parser.add_argument("--server-only-ar-stream-rpc", dest="server_only_ar_turn_rpc", action="store_false", help="For target-side server-only AR, stream one verify_result per generated token.")
    parser.add_argument("--server-only-ar-max-new-tokens", type=int, default=256, help="Maximum new tokens for target-side server-only AR mode.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum new tokens per turn for evaluation generation.")
    parser.add_argument("--bill-draft-as-target-gpu", action="store_true", default=False, help="Bill draft time using target_per_sec_cost (for server-side shared-GPU draft).")
    parser.add_argument("--disable-auto-server-draft-profile", dest="server_draft_profile_auto", action="store_false", default=True, help="Disable target-side auto generation/loading of server draft profile in server-only mode.")
    parser.add_argument("--server-draft-profile-force-refresh", action="store_true", default=False, help="Force target-side server draft profile regeneration in server-only mode.")
    parser.add_argument("--server-draft-profile-model-calls-per-count", type=int, default=100, help="Target-side server draft profile model calls per width.")
    parser.add_argument("--server-draft-profile-width-list", type=str, default="10,20,30,40,50,60,70,80,90,100,110,120,130,140,150", help="Target-side server draft profile width list CSV.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducibility (sets torch/random/numpy seeds when specified)")
    parser.add_argument("--deterministic", action="store_true", default=False, help="Ensure identical results including GPU operations (PyTorch deterministic mode; some operations may be slower)")
    args = parser.parse_args()
    if float(args.draft_electricity_cost_per_kwh) < 0:
        raise ValueError("--draft-electricity-cost-per-kwh must be >= 0.")

    if args.question_file:
        question_file = args.question_file
    else:
        script_dir = os.path.dirname(__file__)
        parent_dir = os.path.dirname(script_dir)
        if args.bench_name == "mt_bench":
            question_file = f"{parent_dir}/data/mt_bench/question.jsonl"
        elif args.bench_name == "gsm8k":
            question_file = f"{parent_dir}/data/gsm8k.jsonl"
        else:
            # HF benchmark
            question_file = ""

    # answer_file
    if args.answer_file is None:
        script_dir = os.path.dirname(__file__)
        parent_dir = os.path.dirname(script_dir)
        args.answer_file = f"{parent_dir}/result/draft_answers_{args.bench_name}.jsonl"
    
    # model_id temperature 
    args.model_id = args.model_id + "-temperature-" + str(args.temperature)

    if args.fixed_width_value is not None:
        args.fixed_width_value = max(1, int(args.fixed_width_value))

    # Quantization default policy:
    # - If user did not pass --load-in-4bit/8bit:
    # * 8B draft models -> 4bit (OOM )
    #   * 7B draft models -> 8bit
    #   * others -> none
    # - Otherwise keep explicit user choice.
    if (not bool(args.load_in_4bit)) and (not bool(args.load_in_8bit)):
        if _draft_model_prefers_default_4bit(args.draft_model_path):
            args.load_in_4bit = True
            print(
                "[INFO] draft quantization default applied: "
                f"{args.draft_model_path} -> 4bit"
            )
        elif _draft_model_prefers_default_8bit(args.draft_model_path):
            args.load_in_8bit = True
            print(
                "[INFO] draft quantization default applied: "
                f"{args.draft_model_path} -> 8bit"
            )
    target_quantization_raw = str(args.target_quantization or "auto").strip().lower()
    if target_quantization_raw == "auto":
        if _target_model_prefers_default_8bit(args.base_model_path):
            args.target_quantization = "8bit"
        else:
            args.target_quantization = "none"
        print(
            "[INFO] target quantization default applied: "
            f"{args.base_model_path} -> {args.target_quantization}"
        )
    else:
        args.target_quantization = _normalize_quantization_mode(target_quantization_raw, default="none")

    if args.seed is not None:
        set_seed(args.seed)
        print(f"[draft] seed fixed: {args.seed}")
    draft_profile_model_calls_per_count = max(1, int(args.draft_profile_model_calls_per_count))
    target_profile_model_calls_per_count = max(1, int(args.target_profile_model_calls_per_count))
    auto_profile = bool(args.auto_profile) and (not bool(args.disable_auto_profile))
    draft_profile_force_refresh = bool(args.draft_profile_force_refresh or args.force_profile_refresh)
    target_profile_force_refresh = bool(args.target_profile_force_refresh or args.force_profile_refresh)
    online_profile_update = not bool(args.disable_online_profile_update)
    online_profile_lr = _sanitize_online_lr(args.online_profile_lr)
    accept_length_calibration = not bool(
        args.disable_accept_length_calibration or args.disable_online_calibration
    )
    target_time_calibration = not bool(
        args.disable_target_time_calibration or args.disable_online_calibration
    )
    
    if args.deterministic:
        set_deterministic()
        print("[draft] deterministic mode enabled")
    
    if args.chat_mode:
        run_chat_mode(
            host=args.host,
            port=args.port,
            base_model_path=args.base_model_path,
            draft_model_path=args.draft_model_path,
            bench_name=args.bench_name,
            question_file=question_file,
            temperature=args.temperature,
            nodes=args.nodes,
            max_depth=args.max_depth,
            device_map=args.device_map,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            target_quantization=args.target_quantization,
            per_token_probability_bound=args.per_token_probability_bound,
            per_path_probability_bound=args.per_path_probability_bound,
            device_name=args.device_name,
            draft_per_hour_cost=args.draft_per_hour_cost,
            target_per_hour_cost=args.target_per_hour_cost,
            draft_electricity_cost_per_kwh=args.draft_electricity_cost_per_kwh,
            user_communication_cost_per_gb=args.user_communication_cost_per_gb,
            cloud_outbound_cost_per_gb=args.cloud_outbound_cost_per_gb,
            accept_length_margin=args.accept_length_margin,
            objective_selection_mode=args.objective_selection_mode,
            constraint_target=args.constraint_target,
            metric_constraint_per_1m_token=args.metric_constraint_per_1m_token,
            min_tps_constraint=args.min_tps_constraint,
            cost_sensitivity=args.cost_sensitivity,
            min_width=args.min_width,
            fixed_depth=args.fixed_depth,
            fixed_nnodes=args.fixed_nnodes,
            fixed_width=args.fixed_width,
            fixed_width_value=args.fixed_width_value,
            server_name=args.server_name,
            enable_gpu_monitor=args.enable_gpu_monitor,
            gpu_monitor_interval=args.gpu_monitor_interval,
            enable_cpu_monitor=args.enable_cpu_monitor,
            fix_gpu_clock=args.fix_gpu_clock,
            gpu_graphics_clock_mhz=args.gpu_graphics_clock_mhz,
            gpu_memory_clock_mhz=args.gpu_memory_clock_mhz,
            objective_metric=args.objective_metric,
            no_draft_cost=args.no_draft_cost,
            tokenizer_path=args.tokenizer_path,
            chat_max_new_tokens=args.chat_max_new_tokens,
            auto_target_profile=auto_profile,
            draft_profile_force_refresh=draft_profile_force_refresh,
            draft_profile_width_list=args.draft_profile_width_list,
            target_profile_force_refresh=target_profile_force_refresh,
            target_profile_model_calls_per_count=target_profile_model_calls_per_count,
            target_profile_width_list="50",
            target_profile_depth_list="10",
            target_profile_node_list=args.target_profile_node_list,
            online_profile_update=online_profile_update,
            online_profile_lr=online_profile_lr,
            bill_draft_as_target_gpu=args.bill_draft_as_target_gpu,
            server_draft_profile_auto=args.server_draft_profile_auto,
            server_draft_profile_force_refresh=args.server_draft_profile_force_refresh,
            server_draft_profile_model_calls_per_count=max(1, int(args.server_draft_profile_model_calls_per_count)),
            server_draft_profile_width_list=args.server_draft_profile_width_list,
        )
    else:
        print(f"Output to {args.answer_file}")
        run_draft(
            host=args.host,
            port=args.port,
            base_model_path=args.base_model_path,
            draft_model_path=args.draft_model_path,
            bench_name=args.bench_name,
            question_file=question_file,
            draft_per_hour_cost=args.draft_per_hour_cost,
            target_per_hour_cost=args.target_per_hour_cost,
            draft_electricity_cost_per_kwh=args.draft_electricity_cost_per_kwh,
            user_communication_cost_per_gb=args.user_communication_cost_per_gb,
            cloud_outbound_cost_per_gb=args.cloud_outbound_cost_per_gb,
            accept_length_margin=args.accept_length_margin,
            objective_selection_mode=args.objective_selection_mode,
            constraint_target=args.constraint_target,
            metric_constraint_per_1m_token=args.metric_constraint_per_1m_token,
            min_tps_constraint=args.min_tps_constraint,
            total_metric_cap=args.total_metric_cap,
            cost_sensitivity=args.cost_sensitivity,
            opt_tree=args.opt_tree,
            limit=args.limit,
            temperature=args.temperature,
            nodes=args.nodes,
            max_depth=args.max_depth,
            device_map=args.device_map,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            target_quantization=args.target_quantization,
            answer_file=args.answer_file,
            model_id=args.model_id,
            num_choices=args.num_choices,
            debug=args.debug,
            print_tree=args.print_tree,
            per_token_probability_bound=args.per_token_probability_bound,
            per_path_probability_bound=args.per_path_probability_bound,
            device_name=args.device_name,
            min_width=args.min_width,
            fixed_depth=args.fixed_depth,
            fixed_nnodes=args.fixed_nnodes,
            fixed_width=args.fixed_width,
            fixed_width_value=args.fixed_width_value,
            server_name=args.server_name,
            enable_gpu_monitor=args.enable_gpu_monitor,
            gpu_monitor_interval=args.gpu_monitor_interval,
            enable_cpu_monitor=args.enable_cpu_monitor,
            fix_gpu_clock=args.fix_gpu_clock,
            gpu_graphics_clock_mhz=args.gpu_graphics_clock_mhz,
            gpu_memory_clock_mhz=args.gpu_memory_clock_mhz,
            proactive_drafting=args.proactive_drafting,
            proactive_threshold=args.proactive_threshold,
            adaptive_proactive_threshold=args.adaptive_proactive_threshold,
            disable_proactive_budget=args.disable_proactive_budget,
            join_canceled_proactive_before_tree_build=args.join_canceled_proactive_before_tree_build,
            no_draft_cost=args.no_draft_cost,
            objective_metric=args.objective_metric,
            server_only_baseline_json=args.server_only_baseline_json,
            disable_server_only=args.disable_server_only,
            force_server_only=args.force_server_only,
            force_server_only_ar=args.force_server_only_ar,
            server_only_ar_turn_rpc=args.server_only_ar_turn_rpc,
            server_only_ar_max_new_tokens=args.server_only_ar_max_new_tokens,
            max_new_tokens=args.max_new_tokens,
            reference_test_mode=args.reference_test_mode,
            reference_test_runs=args.reference_test_runs,
            reference_test_output_json=args.reference_test_output_json,
            reference_cs_curve_rounds=args.reference_cs_curve_rounds,
            reference_max_steps_limit=args.reference_max_steps_limit,
            reference_constraint_multipliers=args.reference_constraint_multipliers,
            reference_force_refresh=args.reference_force_refresh,
            tokenizer_path=args.tokenizer_path,
            auto_target_profile=auto_profile,
            draft_profile_force_refresh=draft_profile_force_refresh,
            draft_profile_model_calls_per_count=draft_profile_model_calls_per_count,
            draft_profile_width_list=args.draft_profile_width_list,
            draft_profile_only=args.draft_profile_only,
            target_profile_force_refresh=target_profile_force_refresh,
            target_profile_model_calls_per_count=target_profile_model_calls_per_count,
            target_profile_width_list="50",
            target_profile_depth_list="10",
            target_profile_node_list=args.target_profile_node_list,
            profile_only=args.profile_only,
            profile_only_report_json=args.profile_only_report_json,
            reference_only_exit_after_cache=args.reference_only_exit_after_cache,
            online_profile_update=online_profile_update,
            online_profile_lr=online_profile_lr,
            accept_length_calibration=accept_length_calibration,
            target_time_calibration=target_time_calibration,
            bill_draft_as_target_gpu=args.bill_draft_as_target_gpu,
            server_draft_profile_auto=args.server_draft_profile_auto,
            server_draft_profile_force_refresh=args.server_draft_profile_force_refresh,
            server_draft_profile_model_calls_per_count=max(1, int(args.server_draft_profile_model_calls_per_count)),
            server_draft_profile_width_list=args.server_draft_profile_width_list,
        )

    # answer_file (reorg_answer_file )
    if (not args.chat_mode) and args.answer_file and os.path.exists(args.answer_file):
        def reorg_answer_file(answer_file):
            """Sort by question id and de-duplication"""
            answers = {}
            with open(answer_file, "r") as fin:
                for l in fin:
                    qid = json.loads(l)["question_id"]
                    answers[qid] = l

            qids = sorted(list(answers.keys()))
            with open(answer_file, "w") as fout:
                for qid in qids:
                    fout.write(answers[qid])
        
        reorg_answer_file(args.answer_file)



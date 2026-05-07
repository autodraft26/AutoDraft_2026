"""
Orin-specific draft execution wrapper.

The latest logic uses `evaluation.eval_autodraft_draft` directly,
and only patches the GPU monitor for the Orin (jtop) environment.
"""

import importlib
import os
import runpy
import sys
import threading
import time
from typing import Any, Dict, List

import opt_classic.utils as classic_utils


class OrinGPUMonitor(classic_utils.GPUMonitor):
    """Orin(jtop) dedicated GPU monitor. In case of failure, error handling occurs immediately without fallback."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backend = "orin_jtop"
        self._jtop_client = None
        self._jtop_class = None
        self._jtop_lock = threading.Lock()
        self._jtop_site_path = os.path.expanduser("~/.local/share/jtop/lib/python3.12/site-packages")
        self._monitor_error = None
        self._jtop_last_error = None
        self._jtop_read_retries = 3
        self._jtop_retry_delay_sec = 0.2

    def __del__(self):
        try:
            self._close_jtop_client()
        except Exception:
            pass

    @staticmethod
    def _to_float(value):
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            # Keep only numeric characters from strings such as "23.5W" or "1200 MHz".
            filtered = "".join(ch for ch in value if (ch.isdigit() or ch in ".-"))
            if not filtered or filtered in {".", "-", "-."}:
                return None
            try:
                return float(filtered)
            except Exception:
                return None
        return None

    @classmethod
    def _search_first_numeric(cls, obj, include_keywords: List[str], exclude_keywords: List[str] = None):
        """
        Find the first numeric value in nested dict/list structures by key name.
        - include_keywords: candidate strings that must be included in the key (lowercase)
        - exclude_keywords: candidate strings that exclude the key when present (lowercase)
        """
        if exclude_keywords is None:
            exclude_keywords = []

        def _walk(node):
            if isinstance(node, dict):
                for k, v in node.items():
                    kl = str(k).lower()
                    if any(ex in kl for ex in exclude_keywords):
                        continue
                    if any(inc in kl for inc in include_keywords):
                        parsed = cls._to_float(v)
                        if parsed is not None:
                            return parsed
                    found = _walk(v)
                    if found is not None:
                        return found
            elif isinstance(node, list):
                for item in node:
                    found = _walk(item)
                    if found is not None:
                        return found
            return None

        return _walk(obj)

    @classmethod
    def _search_max_numeric(cls, obj, include_keywords: List[str], exclude_keywords: List[str] = None):
        """Find the maximum numerical value that meets the key name conditions in the nested structure."""
        if exclude_keywords is None:
            exclude_keywords = []
        values = []

        def _walk(node):
            if isinstance(node, dict):
                for k, v in node.items():
                    kl = str(k).lower()
                    if any(ex in kl for ex in exclude_keywords):                        continue
                    if any(inc in kl for inc in include_keywords):
                        parsed = cls._to_float(v)
                        if parsed is not None:
                            values.append(parsed)
                    _walk(v)
            elif isinstance(node, list):
                for item in node:
                    _walk(item)

        _walk(obj)
        if values:
            return max(values)
        return None

    @classmethod
    def _extract_power_from_rails(cls, power_obj):
        """
        Extract GPU rail power (mW) and limit values (mW) from jtop power.rail.
        Priority: gpu/gr3d rail containing -> vdd_gpu_soc -> other rail/tot.
        """
        if not isinstance(power_obj, dict):
            return None, None

        rail_map = power_obj.get("rail", {})
        selected_power = None
        selected_limit = None

        def _pick_from_rail(rail_data):
            if not isinstance(rail_data, dict):
                return None, None
            p = cls._to_float(rail_data.get("power"))
            if p is None:
                p = cls._to_float(rail_data.get("avg"))
            lim = cls._to_float(rail_data.get("warn"))
            if lim is None:
                lim = cls._to_float(rail_data.get("crit"))
            return p, lim

        # 1) GPU rail
        if isinstance(rail_map, dict):
            for rail_name, rail_data in rail_map.items():
                name = str(rail_name).lower()
                if ("gpu" in name) or ("gr3d" in name):
                    p, lim = _pick_from_rail(rail_data)
                    if p is not None:
                        selected_power = p
                        selected_limit = lim
                        break

            # 2) GPU rail SOC/VDD rail power
            if selected_power is None:
                best_p = None
                best_lim = None
                for rail_name, rail_data in rail_map.items():
                    name = str(rail_name).lower()
                    if ("soc" in name) or ("vdd" in name):
                        p, lim = _pick_from_rail(rail_data)
                        if p is not None and (best_p is None or p > best_p):
                            best_p = p
                            best_lim = lim
                if best_p is not None:
                    selected_power = best_p
                    selected_limit = best_lim

            # 3) rail
            if selected_power is None:
                best_p = None
                best_lim = None
                for _, rail_data in rail_map.items():
                    p, lim = _pick_from_rail(rail_data)
                    if p is not None and (best_p is None or p > best_p):
                        best_p = p
                        best_lim = lim
                if best_p is not None:
                    selected_power = best_p
                    selected_limit = best_lim

        # 4) : total power
        if selected_power is None:
            tot = power_obj.get("tot", {})
            if isinstance(tot, dict):
                selected_power = cls._to_float(tot.get("power"))
                if selected_limit is None:
                    selected_limit = cls._to_float(tot.get("warn")) or cls._to_float(tot.get("crit"))

        return selected_power, selected_limit

    def get_gpu_info(self):
        last_err_detail = None
        for attempt in range(1, int(self._jtop_read_retries) + 1):
            jtop_info = self._get_gpu_info_from_jtop()
            if jtop_info is not None:
                return jtop_info
            last_err_detail = self._jtop_last_error
            # client
            self._close_jtop_client()
            if attempt < int(self._jtop_read_retries):
                time.sleep(float(self._jtop_retry_delay_sec))
        detail = f" Detail: {last_err_detail}" if last_err_detail else ""
        raise RuntimeError(
            "Orin GPUMonitor: jtop metric collection failed after retries. "
            "nvidia-smi fallback is disabled on Jetson Orin."
            + detail
        )

    @staticmethod
    def _purge_jtop_modules():
        remove_keys = [k for k in list(sys.modules.keys()) if k == "jtop" or k.startswith("jtop.")]
        for key in remove_keys:
            sys.modules.pop(key, None)

    def _import_jtop_class(self):
        import_candidates = [None]
        if os.path.isdir(self._jtop_site_path):
            import_candidates.append(self._jtop_site_path)

        for site_path in import_candidates:
            try:
                if site_path:
                    if site_path not in sys.path:
                        sys.path.insert(0, site_path)
                self._purge_jtop_modules()
                from jtop import jtop  # type: ignore
                return jtop
            except Exception:
                continue
        return None

    def _ensure_jtop_client(self):
        with self._jtop_lock:
            if self._jtop_client is not None:
                try:
                    if self._jtop_client.ok():
                        return True
                except Exception:
                    pass
                self._close_jtop_client_locked()

            if self._jtop_class is None:
                self._jtop_class = self._import_jtop_class()
                if self._jtop_class is None:
                    self._jtop_last_error = "failed to import jtop client module"
                    return False

            try:
                client = self._jtop_class()
                client.start()
                if not client.ok():
                    try:
                        client.close()
                    except Exception:
                        pass
                    self._jtop_last_error = "jtop client started but jetson.ok() is False"
                    return False
                self._jtop_client = client
                self._jtop_last_error = None
                return True
            except Exception as e:
                self._jtop_last_error = repr(e)
                return False

    def _close_jtop_client_locked(self):
        if self._jtop_client is not None:
            try:
                self._jtop_client.close()
            except Exception:
                pass
        self._jtop_client = None

    def _close_jtop_client(self):
        with self._jtop_lock:
            self._close_jtop_client_locked()

    def stop_monitoring(self):
        super().stop_monitoring()
        # step start/stop client .
        # ( _ensure_jtop_client )
        if self._monitor_error is not None:
            err = self._monitor_error
            self._monitor_error = None
            with self._lock:
                has_samples = bool(self.data)
            if not has_samples:
                raise RuntimeError(
                    "Orin GPUMonitor: jtop monitor loop failed. "
                    "nvidia-smi fallback is disabled."
                ) from err

    def start_monitoring(self):
        self._monitor_error = None
        super().start_monitoring()

    def monitor_loop(self):
        """When an error occurs, jtop stops immediately and stores the error so that it can be terminated in the upper logic."""
        while self.monitoring:
            try:
                self._sample_once()
                time.sleep(self.interval)
            except Exception as e:
                with self._lock:
                    has_samples = bool(self.data)
                if not has_samples:
                    self._monitor_error = e
                    self.monitoring = False
                    break
                time.sleep(self.interval)

    def _get_gpu_info_from_jtop(self):
        if not self._ensure_jtop_client():
            return None

        try:
            jetson = self._jtop_client
            if jetson is None or not jetson.ok():
                return None

            # RAM shared GPU (Orin )
            memory_info: Dict[str, Any] = jetson.memory.get("RAM", {})
            memory_used_mb = float(memory_info.get("shared", 0)) / 1000.0
            memory_total_mb = float(memory_info.get("tot", 1)) / 1000.0
            if memory_total_mb <= 0:
                memory_total_mb = 1.0

            gpu_root = jetson.gpu.get("gpu", {})
            gpu_status: Dict[str, Any] = gpu_root.get("status", {})
            gpu_freq: Dict[str, Any] = gpu_root.get("freq", {})
            util = float(gpu_status.get("load", 0))
            graphics_clock = float(gpu_freq.get("cur", 0)) / 1000.0
            max_graphics_clock = float(gpu_freq.get("max", 0)) / 1000.0
            # Orin memory clock GPU
            memory_clock = self._to_float(gpu_freq.get("mem")) or self._to_float(gpu_freq.get("memory"))
            if memory_clock is not None and memory_clock > 20000:
                # Hz MHz
                memory_clock = memory_clock / 1_000_000.0
            max_memory_clock = self._to_float(gpu_freq.get("max_mem")) or self._to_float(gpu_freq.get("memory_max"))
            if max_memory_clock is not None and max_memory_clock > 20000:
                max_memory_clock = max_memory_clock / 1_000_000.0

            temp_obj = jetson.temperature.get("gpu", {})
            if isinstance(temp_obj, dict):
                temperature = float(temp_obj.get("temp", 0))
            else:
                temperature = float(temp_obj or 0)

            # jtop power GPU rail
            # ( key naming )
            power_obj = getattr(jetson, "power", {}) or {}
            power_draw, power_limit = self._extract_power_from_rails(power_obj)

            # rail
            if power_draw is None:
                power_draw = self._search_first_numeric(
                    power_obj,
                    include_keywords=["gpu", "gr3d", "cv", "vdd_gpu", "vdd", "rail"],
                    exclude_keywords=["limit", "cap", "max", "min", "avg"],
                )
            # , gpu/gr3d
            if power_draw is None:
                power_draw = self._search_first_numeric(
                    power_obj,
                    include_keywords=["gpu", "gr3d"],
                )
            if power_limit is None:
                power_limit = self._search_first_numeric(
                    power_obj,
                    include_keywords=["limit", "cap", "warn", "crit"],
                )
            if power_limit is None:
                # power
                power_limit = self._search_max_numeric(
                    power_obj,
                    include_keywords=["power", "w", "mw", "limit", "cap", "gpu", "gr3d"],
                )

            # mW
            if power_draw is not None and power_draw > 1000:
                power_draw = power_draw / 1000.0
            if power_limit is not None and power_limit > 1000:
                power_limit = power_limit / 1000.0
            power_usage_percent = None
            if power_draw is not None and power_limit is not None and power_limit > 0:
                power_usage_percent = (power_draw / power_limit) * 100.0

            now = time.time()
            gpu_info: List[Dict[str, Any]] = [{
                "gpu_id": 0,
                "memory_used_mb": memory_used_mb,
                "memory_total_mb": memory_total_mb,
                "utilization_percent": util,
                "temperature_c": temperature,
                "power_draw_w": power_draw,
                "power_limit_w": power_limit,
                "graphics_clock_mhz": graphics_clock,
                "memory_clock_mhz": memory_clock,
                "max_graphics_clock_mhz": max_graphics_clock if max_graphics_clock > 0 else None,
                "max_memory_clock_mhz": max_memory_clock if (max_memory_clock is not None and max_memory_clock > 0) else None,
                "memory_used_percent": (memory_used_mb / memory_total_mb) * 100.0,
                "power_usage_percent": power_usage_percent,
                "timestamp": now,
            }]
            return gpu_info
        except Exception:
            self._jtop_last_error = "failed while reading jtop metrics"
            self._close_jtop_client()
            return None

    def _append_gpu_info(self, gpu_info, timestamp=None):
        """Record jtop samples using the base GPUMonitor data layout."""
        if not gpu_info:
            return
        ts = float(timestamp if timestamp is not None else time.time())
        new_entries = []
        new_timestamps = []
        for i, gpu in enumerate(gpu_info):
            entry = {
                "timestamp": ts,
                "gpu_id": i,
                **gpu,
            }
            entry["timestamp"] = ts
            new_entries.append(entry)
            new_timestamps.append(ts)
        with self._lock:
            self.data.extend(new_entries)
            self._timestamps.extend(new_timestamps)
            self._prune_locked(time.time() - self.retention_seconds)

    def _sample_once(self):
        """Take one asynchronous-loop jtop sample."""
        gpu_info = self.get_gpu_info()
        self._append_gpu_info(gpu_info)
        return gpu_info


def _patch_gpu_monitor():
    """Replace GPUMonitor with the Orin version before importing the latest draft module."""
    classic_utils.GPUMonitor = OrinGPUMonitor


_DRAFT_MODULE = None


def _get_draft_module():
    """Lazy loading of draft modules (keeping OrinGPUMonitor sole import light)."""
    global _DRAFT_MODULE
    if _DRAFT_MODULE is None:
        _patch_gpu_monitor()
        _DRAFT_MODULE = importlib.import_module("evaluation.eval_autodraft_draft")
    return _DRAFT_MODULE


def __getattr__(name):
    """
    `from evaluation.eval_autodraft_draft_orin import run_draft` compatibility is preserved.
    Load the draft module only when required symbols are accessed.
    """
    if name.startswith("_"):
        raise AttributeError(name)
    draft_module = _get_draft_module()
    if hasattr(draft_module, name):
        value = getattr(draft_module, name)
        globals()[name] = value
        return value
    raise AttributeError(name)


def _expose_draft_symbols():
    """Option: draft Used when you want to immediately deploy module symbols."""
    _patch_gpu_monitor()
    draft_module = importlib.import_module("evaluation.eval_autodraft_draft")
    for name in dir(draft_module):
        if name.startswith("_"):
            continue
        globals()[name] = getattr(draft_module, name)


if __name__ == "__main__":
    _patch_gpu_monitor()
    # draft , GPUMonitor Orin .
    runpy.run_module("evaluation.eval_autodraft_draft", run_name="__main__")

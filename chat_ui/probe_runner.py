import json
import socket
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse
from dataclasses import dataclass
from typing import Dict, Optional

from transformers import AutoTokenizer

try:
    from .server_registry import ModelSpec, ServerSpec
except Exception:
    from server_registry import ModelSpec, ServerSpec

try:
    from opt_classic.utils import recv_json_with_size, send_json_with_size
except Exception:
    from ..opt_classic.utils import recv_json_with_size, send_json_with_size


@dataclass
class ProbeRequest:
    prompt: str = "Summarize speculative decoding in one sentence."
    max_tokens: int = 64
    temperature: float = 0.0
    timeout_s: float = 25.0
    warmup_runs: int = 1
    measured_runs: int = 1


class ProbeRunner:
    def _parse_host_port(self, endpoint: str) -> tuple[str, int]:
        raw = str(endpoint or "").strip()
        if not raw:
            return "127.0.0.1", 26101
        if "://" in raw:
            p = urlparse(raw)
            return str(p.hostname or "127.0.0.1"), int(p.port or 26101)
        part = raw.split("/", 1)[0]
        if ":" in part:
            h, ps = part.rsplit(":", 1)
            try:
                return h.strip(), int(ps.strip())
            except Exception:
                return part.strip(), 26101
        return part, 26101

    def _make_openai_payload(self, model_id: str, req: ProbeRequest) -> Dict:
        return {
            "model": model_id,
            "messages": [{"role": "user", "content": req.prompt}],
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
        }

    def _post_json(self, url: str, payload: Dict, api_key: Optional[str], timeout_s: float) -> Dict:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
            return json.loads(raw) if raw else {}

    def _estimate_cost_per_1m(self, usage: Dict, model: ModelSpec) -> Optional[float]:
        if model.pricing_input_per_1m is None and model.pricing_output_per_1m is None:
            return None
        prompt_tokens = float(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = float(usage.get("completion_tokens", 0) or 0)
        total_tokens = prompt_tokens + completion_tokens
        if total_tokens <= 0:
            return None
        prompt_cost = (prompt_tokens / 1_000_000.0) * float(model.pricing_input_per_1m or 0.0)
        completion_cost = (completion_tokens / 1_000_000.0) * float(model.pricing_output_per_1m or 0.0)
        total_cost = prompt_cost + completion_cost
        return total_cost * (1_000_000.0 / total_tokens)

    def _estimate_energy_per_1m(self, tps: float, server: ServerSpec) -> Optional[float]:
        # Optional estimate for local targets when user supplies metadata.power_watts.
        if server.server_type != "local":
            return None
        power_watts = server.metadata.get("power_watts")
        if power_watts is None:
            return None
        try:
            p = float(power_watts)
            if p <= 0 or tps <= 0:
                return None
            seconds_for_1m = 1_000_000.0 / tps
            kwh = (p * seconds_for_1m) / 3_600_000.0
            return kwh
        except Exception:
            return None

    def probe_server_model(
        self, server: ServerSpec, model: ModelSpec, req: Optional[ProbeRequest] = None
    ) -> Dict:
        request = req or ProbeRequest()
        if not server.enabled:
            return {"ok": False, "error": "server_disabled"}
        if server.requires_api_key and not server.api_key:
            return {"ok": False, "error": "api_key_required"}
        if not server.endpoint:
            return {"ok": False, "error": "missing_endpoint"}
        if server.protocol == "autodraft_target":
            return self._probe_autodraft_target(server, model, request)
        if server.protocol != "openai_chat_completions":
            return {"ok": False, "error": f"unsupported_protocol:{server.protocol}"}

        for _ in range(max(0, request.warmup_runs)):
            try:
                payload = self._make_openai_payload(model.model_id, request)
                self._post_json(server.endpoint, payload, server.api_key, request.timeout_s)
            except Exception:
                pass

        latencies = []
        tps_values = []
        metric_per_1m_values = []
        energy_per_1m_values = []
        last_usage = {}
        last_error = None

        for _ in range(max(1, request.measured_runs)):
            try:
                payload = self._make_openai_payload(model.model_id, request)
                t0 = time.perf_counter()
                out = self._post_json(server.endpoint, payload, server.api_key, request.timeout_s)
                elapsed = max(1e-6, time.perf_counter() - t0)
                usage = out.get("usage", {}) if isinstance(out, dict) else {}
                last_usage = usage if isinstance(usage, dict) else {}
                completion_tokens = float(last_usage.get("completion_tokens", 0) or 0)
                prompt_tokens = float(last_usage.get("prompt_tokens", 0) or 0)
                total_tokens = completion_tokens + prompt_tokens
                if completion_tokens <= 0 and total_tokens > 0:
                    completion_tokens = total_tokens
                tps = completion_tokens / elapsed if completion_tokens > 0 else 0.0
                latencies.append(elapsed)
                tps_values.append(tps)

                est_cost_per_1m = self._estimate_cost_per_1m(last_usage, model)
                if est_cost_per_1m is not None:
                    metric_per_1m_values.append(est_cost_per_1m)
                est_energy_per_1m = self._estimate_energy_per_1m(max(tps, 1e-9), server)
                if est_energy_per_1m is not None:
                    energy_per_1m_values.append(est_energy_per_1m)
            except urllib.error.HTTPError as http_err:
                try:
                    detail = http_err.read().decode("utf-8", errors="ignore")
                except Exception:
                    detail = str(http_err)
                last_error = f"http_error:{http_err.code}:{detail[:220]}"
            except Exception as exc:
                last_error = f"probe_error:{exc}"

        if not latencies:
            return {
                "ok": False,
                "error": last_error or "probe_failed",
                "server_id": server.server_id,
                "model_id": model.model_id,
            }

        avg_latency = sum(latencies) / len(latencies)
        avg_tps = sum(tps_values) / len(tps_values) if tps_values else 0.0
        avg_metric_per_1m = (
            sum(metric_per_1m_values) / len(metric_per_1m_values) if metric_per_1m_values else None
        )
        avg_energy_per_1m = (
            sum(energy_per_1m_values) / len(energy_per_1m_values) if energy_per_1m_values else None
        )
        return {
            "ok": True,
            "server_id": server.server_id,
            "server_name": server.name,
            "server_type": server.server_type,
            "model_id": model.model_id,
            "model_label": model.label,
            "latency_s": avg_latency,
            "throughput_tps": avg_tps,
            "metric_per_1m": avg_metric_per_1m,
            "target_energy_per_1m_kwh": avg_energy_per_1m,
            "draft_energy_per_1m_kwh": None,
            "raw_usage": last_usage,
            "timestamp": time.time(),
            "probe_mode": "server_only_api",
        }

    def _probe_autodraft_target(self, server: ServerSpec, model: ModelSpec, request: ProbeRequest) -> Dict:
        host, port = self._parse_host_port(server.endpoint)
        tokenizer_source = (
            str(server.metadata.get("tokenizer_path"))
            if isinstance(server.metadata, dict) and server.metadata.get("tokenizer_path")
            else str(model.model_id)
        )
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        except Exception:
            # Fallback to local default tokenizer when model id is remote-only alias.
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")

        prompt_ids = tokenizer.encode(str(request.prompt), add_special_tokens=False)
        latencies = []
        tps_values = []
        metric_per_1m_values = []
        energy_per_1m_values = []
        last_error = None

        for _ in range(max(0, request.warmup_runs)):
            try:
                with socket.create_connection((host, int(port)), timeout=float(request.timeout_s)) as sock:
                    send_json_with_size(sock, {"type": "status"})
                    _ = recv_json_with_size(sock)
                    send_json_with_size(sock, {"type": "reload_model", "base_model_path": str(model.model_id), "quantization": "8bit"})
                    _ = recv_json_with_size(sock)
                    send_json_with_size(sock, {"type": "init", "input_ids": prompt_ids})
                    init_reply, _ = recv_json_with_size(sock)
                    next_token = int(init_reply.get("next_token", 0))
                    send_json_with_size(
                        sock,
                        {
                            "type": "tree_step",
                            "draft_input_ids": [next_token],
                            "draft_position_ids": [0],
                            "tree_attention_mask": [[1]],
                            "parent": [0],
                        },
                    )
                    _ = recv_json_with_size(sock)
            except Exception:
                pass

        for _ in range(max(1, request.measured_runs)):
            try:
                t0 = time.perf_counter()
                generated = 0
                with socket.create_connection((host, int(port)), timeout=float(request.timeout_s)) as sock:
                    send_json_with_size(sock, {"type": "status"})
                    _ = recv_json_with_size(sock)
                    send_json_with_size(sock, {"type": "reload_model", "base_model_path": str(model.model_id), "quantization": "8bit"})
                    _ = recv_json_with_size(sock)
                    send_json_with_size(sock, {"type": "init", "input_ids": prompt_ids})
                    init_reply, _ = recv_json_with_size(sock)
                    next_token = int(init_reply.get("next_token", 0))
                    for _step in range(max(1, int(request.max_tokens))):
                        send_json_with_size(
                            sock,
                            {
                                "type": "tree_step",
                                "draft_input_ids": [next_token],
                                "draft_position_ids": [0],
                                "tree_attention_mask": [[1]],
                                "parent": [0],
                            },
                        )
                        reply, _ = recv_json_with_size(sock)
                        if str(reply.get("type")) != "verify_result":
                            raise RuntimeError(f"invalid_reply:{reply}")
                        generated += int(reply.get("accept_length", 0)) + 1
                        next_token = int(reply.get("next_token", next_token))
                        if bool(reply.get("eos_reached", False)):
                            break
                elapsed = max(1e-6, time.perf_counter() - t0)
                tps = float(generated) / elapsed if generated > 0 else 0.0
                latencies.append(elapsed)
                tps_values.append(tps)

                # Optional rough metric estimate from known pricing.
                total_tokens = float(len(prompt_ids) + max(0, generated))
                if total_tokens > 0 and (
                    model.pricing_input_per_1m is not None or model.pricing_output_per_1m is not None
                ):
                    prompt_cost = (float(len(prompt_ids)) / 1_000_000.0) * float(model.pricing_input_per_1m or 0.0)
                    comp_cost = (float(generated) / 1_000_000.0) * float(model.pricing_output_per_1m or 0.0)
                    total_cost = prompt_cost + comp_cost
                    metric_per_1m_values.append(total_cost * (1_000_000.0 / total_tokens))
                est_energy_per_1m = self._estimate_energy_per_1m(max(tps, 1e-9), server)
                if est_energy_per_1m is not None:
                    energy_per_1m_values.append(est_energy_per_1m)
            except Exception as exc:
                last_error = f"bridge_probe_error:{exc}"

        if not latencies:
            return {
                "ok": False,
                "error": last_error or "probe_failed",
                "server_id": server.server_id,
                "model_id": model.model_id,
            }

        avg_latency = sum(latencies) / len(latencies)
        avg_tps = sum(tps_values) / len(tps_values) if tps_values else 0.0
        avg_metric_per_1m = (
            sum(metric_per_1m_values) / len(metric_per_1m_values) if metric_per_1m_values else None
        )
        avg_energy_per_1m = (
            sum(energy_per_1m_values) / len(energy_per_1m_values) if energy_per_1m_values else None
        )
        return {
            "ok": True,
            "server_id": server.server_id,
            "server_name": server.name,
            "server_type": server.server_type,
            "model_id": model.model_id,
            "model_label": model.label,
            "latency_s": avg_latency,
            "throughput_tps": avg_tps,
            "metric_per_1m": avg_metric_per_1m,
            "target_energy_per_1m_kwh": avg_energy_per_1m,
            "draft_energy_per_1m_kwh": None,
            "raw_usage": {},
            "timestamp": time.time(),
            "probe_mode": "hybrid_capable_target",
        }

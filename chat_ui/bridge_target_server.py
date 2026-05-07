import argparse
import json
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional

from transformers import AutoTokenizer

try:
    from opt_classic.utils import recv_json, send_json
except Exception:
    # Fallback for alternative import paths.
    from ..opt_classic.utils import recv_json, send_json


@dataclass
class BridgeConfig:
    endpoint: str
    api_key: Optional[str]
    external_model: str
    tokenizer_path: str
    temperature: float = 0.0
    timeout_s: float = 30.0
    max_tokens_per_step: int = 1
    model_map: Dict[str, str] = None


class BridgeSession:
    def __init__(self, cfg: BridgeConfig, tokenizer: AutoTokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.current_input_ids: List[int] = []
        self.loaded_model: str = cfg.external_model
        self.quantization: str = "api"

    def status_payload(self) -> Dict:
        return {
            "type": "status_ok",
            "loaded_model": self.loaded_model,
            "quantization": self.quantization,
            "device_map": "external_api",
            "int8_cpu_offload": False,
        }

    def reload_model(self, base_model_path: str, quantization: str) -> Dict:
        mapped = self.cfg.model_map.get(base_model_path, base_model_path)
        self.loaded_model = mapped or self.cfg.external_model
        self.quantization = str(quantization or "api")
        return {
            "type": "reload_ok",
            "loaded_model": self.loaded_model,
            "quantization": self.quantization,
            "device_map": "external_api",
            "int8_cpu_offload": False,
        }

    def init(self, input_ids: List[int]) -> Dict:
        self.current_input_ids = [int(x) for x in (input_ids or [])]
        next_token = self._predict_next_token(self.current_input_ids)
        return {"type": "init_ok", "next_token": int(next_token)}

    def tree_step(self, payload: Dict) -> Dict:
        draft_input_ids = payload.get("draft_input_ids") or []
        if not isinstance(draft_input_ids, list) or not draft_input_ids:
            return {"type": "error", "message": "invalid draft_input_ids"}
        head_token = int(draft_input_ids[0])

        t0 = time.time()
        # Minimal safe contract: accept only the head node.
        accepted_tokens = [head_token]
        best_ids = [0]
        accept_length = 0
        next_token = self._predict_next_token(self.current_input_ids + accepted_tokens)
        self.current_input_ids.extend(accepted_tokens)
        eos_id = self.tokenizer.eos_token_id
        eos_reached = bool(eos_id in accepted_tokens) if eos_id is not None else False
        t1 = time.time()

        return {
            "type": "verify_result",
            "accepted_tokens": accepted_tokens,
            "accept_length": int(accept_length),
            "next_token": int(next_token),
            "eos_reached": bool(eos_reached),
            "best_ids": best_ids,
            "target_verification_time_ms": max(0.0, (t1 - t0) * 1000.0),
            "target_recv_end_time": t1,
            "target_send_start_time": t1,
            "timing_stats": {
                "total_time_seconds": max(0.0, t1 - t0),
                "target_verification_time_ms": max(0.0, (t1 - t0) * 1000.0),
                "timestamp": t1,
            },
            "tree_build_time_ms": 0.0,
            "draft_to_target_time_ms": 0.0,
            "target_to_draft_time_ms": 0.0,
            "final_nnodes": int(len(draft_input_ids)),
            "tree_depth": 1,
            "depth_widths": [int(max(1, len(draft_input_ids)))],
        }

    def _predict_next_token(self, input_ids: List[int]) -> int:
        if not input_ids:
            return int(self.tokenizer.eos_token_id or 0)
        prompt = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        payload = {
            "model": self.loaded_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(self.cfg.temperature),
            "max_tokens": int(self.cfg.max_tokens_per_step),
        }
        out = self._post_json(payload)
        text = self._extract_text(out)
        if not text:
            return int(self.tokenizer.eos_token_id or 0)
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            return int(self.tokenizer.eos_token_id or 0)
        return int(ids[0])

    def _post_json(self, payload: Dict) -> Dict:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        req = urllib.request.Request(
            url=self.cfg.endpoint,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=float(self.cfg.timeout_s)) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8", errors="ignore")
            except Exception:
                detail = str(e)
            raise RuntimeError(f"bridge_http_error:{e.code}:{detail[:220]}") from e
        except Exception as e:
            raise RuntimeError(f"bridge_request_failed:{e}") from e

    @staticmethod
    def _extract_text(out: Dict) -> str:
        try:
            choices = out.get("choices") or []
            if not choices:
                return ""
            c0 = choices[0] or {}
            msg = c0.get("message") or {}
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str):
                    return content
            txt = c0.get("text")
            return txt if isinstance(txt, str) else ""
        except Exception:
            return ""


def _parse_model_map(raw: str) -> Dict[str, str]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        return {}
    return {}


def serve(host: str, port: int, cfg: BridgeConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
    session = BridgeSession(cfg, tokenizer)
    print(f"[bridge] listening on {host}:{port}, endpoint={cfg.endpoint}, model={cfg.external_model}")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen()
        while True:
            conn, addr = s.accept()
            print(f"[bridge] client connected: {addr}")
            with conn:
                while True:
                    try:
                        msg = recv_json(conn)
                    except ConnectionError:
                        break
                    except Exception as e:
                        send_json(conn, {"type": "error", "message": f"invalid_json:{e}"})
                        continue
                    mtype = str(msg.get("type", "")).lower()
                    try:
                        if mtype == "status":
                            send_json(conn, session.status_payload())
                            continue
                        if mtype == "reload_model":
                            send_json(
                                conn,
                                session.reload_model(
                                    str(msg.get("base_model_path", "")),
                                    str(msg.get("quantization", "api")),
                                ),
                            )
                            continue
                        if mtype == "init":
                            send_json(conn, session.init(msg.get("input_ids") or []))
                            continue
                        if mtype == "tree_step":
                            send_json(conn, session.tree_step(msg))
                            continue
                        if mtype in {"shutdown", "stop"}:
                            send_json(conn, {"type": "shutdown_ok"})
                            break
                        # Server-only is not supported in bridge runtime.
                        if mtype.startswith("server_only"):
                            send_json(conn, {"type": "error", "message": "server_only_not_supported"})
                            continue
                        send_json(conn, {"type": "error", "message": f"unknown_type:{mtype}"})
                    except Exception as e:
                        send_json(conn, {"type": "error", "message": str(e)})
            print("[bridge] client disconnected")


def main():
    parser = argparse.ArgumentParser(description="AutoDraft external API bridge target server")
    parser.add_argument("--host", type=str, default=os.environ.get("AUTODRAFT_BRIDGE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("AUTODRAFT_BRIDGE_PORT", "26101")))
    parser.add_argument("--endpoint", type=str, default=os.environ.get("AUTODRAFT_BRIDGE_ENDPOINT", ""))
    parser.add_argument("--api-key", type=str, default=os.environ.get("AUTODRAFT_BRIDGE_API_KEY", ""))
    parser.add_argument("--external-model", type=str, default=os.environ.get("AUTODRAFT_BRIDGE_MODEL", ""))
    parser.add_argument("--tokenizer-path", type=str, default=os.environ.get("AUTODRAFT_BRIDGE_TOKENIZER", "meta-llama/Llama-3.3-70B-Instruct"))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("AUTODRAFT_BRIDGE_TEMPERATURE", "0.0")))
    parser.add_argument("--timeout-s", type=float, default=float(os.environ.get("AUTODRAFT_BRIDGE_TIMEOUT_S", "30.0")))
    parser.add_argument("--max-tokens-per-step", type=int, default=int(os.environ.get("AUTODRAFT_BRIDGE_MAX_TOKENS_PER_STEP", "1")))
    parser.add_argument("--model-map-json", type=str, default=os.environ.get("AUTODRAFT_BRIDGE_MODEL_MAP_JSON", ""))
    args = parser.parse_args()

    if not args.endpoint:
        raise RuntimeError("Bridge endpoint is required (--endpoint or AUTODRAFT_BRIDGE_ENDPOINT).")
    if not args.external_model:
        raise RuntimeError("Bridge external model is required (--external-model or AUTODRAFT_BRIDGE_MODEL).")

    cfg = BridgeConfig(
        endpoint=str(args.endpoint).rstrip("/"),
        api_key=(str(args.api_key).strip() or None),
        external_model=str(args.external_model),
        tokenizer_path=str(args.tokenizer_path),
        temperature=float(args.temperature),
        timeout_s=float(args.timeout_s),
        max_tokens_per_step=max(1, int(args.max_tokens_per_step)),
        model_map=_parse_model_map(str(args.model_map_json)),
    )
    serve(str(args.host), int(args.port), cfg)


if __name__ == "__main__":
    main()

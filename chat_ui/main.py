from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState
from pathlib import Path
import json
import asyncio
import os
try:
    from .autodraft_service import AutoDraftService
except Exception:
    from autodraft_service import AutoDraftService

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent


def _load_dotenv(dotenv_path: Path):
    """Lightweight .env loader without external dependency."""
    if not dotenv_path.exists():
        return
    try:
        for raw in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            if not key:
                continue
            val = val.strip()
            if len(val) >= 2 and ((val[0] == '"' and val[-1] == '"') or (val[0] == "'" and val[-1] == "'")):
                val = val[1:-1]
            os.environ.setdefault(key, val)
    except Exception:
        # Keep app boot resilient even if .env is malformed.
        return


_load_dotenv(REPO_ROOT / ".env")

app = FastAPI()
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
service = AutoDraftService(REPO_ROOT)


@app.get("/")
async def root():
    return FileResponse(BASE_DIR / "static" / "index.html")


def _help_text(mode: str) -> str:
    if mode == "chat":
        return (
            "Chat Mode Help\n"
            "\n"
            "[Basic Usage]\n"
            "- Type a prompt to generate replies with hybrid speculative decoding.\n"
            "- Use Start/Stop to start or stop the chat session.\n"
            "- In autodraft_target mode: Start loads selected target model/quantization, Stop requests target unload.\n"
            "- Use Shutdown Target to terminate the remote autodraft_target process.\n"
            "- Enter `help` or `/help` to show this message again.\n"
            "\n"
            "[Server Add / Selection]\n"
            "- Choose available servers from `Server`.\n"
            "- Server candidates refresh automatically after Add Server or Set Key.\n"
            "- Add a custom server via `Add Server (Manual)`.\n"
            "  - Server Name: display label\n"
            "  - IP Address: target host/IP\n"
            "  - Port: target socket port\n"
            "- Added servers are registered as `autodraft_target` socket servers.\n"
            "- Hybrid-capable algorithms require `autodraft_target` protocol with verify/logits/KV signals.\n"
            "\n"
            "[Model and Recommendation]\n"
            "- Choose the active model in `Server Model`.\n"
            "- Choose recommendation objective in `Metric Preference` (cost/energy).\n"
            "- `Profile Servers` probes server-model candidates and computes recommendations.\n"
            "- Probe labels: `Server-only API` (openai_chat_completions), `Hybrid-capable Target` (autodraft_target).\n"
            "- Recommendation tags:\n"
            "  - Best Efficiency: highest throughput per selected metric\n"
            "  - Pareto-Optimal: non-dominated candidates\n"
            "  - Fastest: highest throughput\n"
            "- Clicking points on the trade-off chart syncs server/model selection.\n"
            "\n"
            "[Runtime Tuning]\n"
            "- Objective Selection: balanced(blend) / constraint\n"
            "- Use Cost Sensitivity or Constraint slider to adjust trade-off behavior.\n"
            "- `Refresh Cache` recomputes/reloads reference cache.\n"
            "\n"
            "[Troubleshooting]\n"
            "- If probing fails, verify endpoint, model ID, API key, and network reachability.\n"
            "- Servers missing required API keys may be excluded from probing/recommendation.\n"
            "- If the chart is empty, rerun reference cache refresh or server profiling."
        )
    return (
        "Benchmark Mode Help\n"
        "- Start: run benchmark\n"
        "- Stop: stop benchmark\n"
        "- In autodraft_target mode: Start loads selected target model/quantization, Stop requests target unload.\n"
        "- Shutdown Target: terminate remote autodraft_target process.\n"
        "- `status` or `/status`: print runtime status/metrics\n"
        "- `help` or `/help`: show this help message\n"
        "- External Chat API runs benchmark in Server-only AR mode only.\n"
        "- Server/model/recommendation settings are shared with Chat mode."
    )


def _is_ws_closed_error(exc: Exception) -> bool:
    m = str(exc or "").lower()
    return (
        "close message has been sent" in m
        or "cannot call \"send\"" in m
        or "cannot call \"receive\"" in m
        or "websocket is not connected" in m
        or "disconnect" in m
    )


async def _safe_send_json(ws: WebSocket, payload: dict) -> bool:
    if ws.client_state != WebSocketState.CONNECTED:
        return False
    try:
        await ws.send_text(json.dumps(payload))
        return True
    except Exception as exc:
        if _is_ws_closed_error(exc):
            return False
        raise


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()

    history = []
    settings = {
        "server": "RTX Pro 6000",
        "selected_server_id": None,
        "selected_model_id": None,
        "target_quantization": "none",
        "draft_quantization": "none",
        "cost": 0.15,
        "max_new_tokens": 512,
        "objective_selection_mode": "blend",
        "metric_preference": "total_cost",
        "constraint_target": "metric",
        "metric_constraint_per_1m_token": 14.0,
        "min_tps_constraint": 0.0,
        "algorithm": "AutoDraft",
        "draft_model_path": "meta-llama/Llama-3.2-3B-Instruct",
        "benchmark_dataset": "mt_bench",
        "proactive_drafting": False,
        "online_profile_update": True,
        "online_profile_lr": 0.05,
        "mode": "chat",
        "chat_active": True,
    }
    
    # Stats
    stats_task = asyncio.create_task(send_stats_periodically(ws))
    chat_task = None

    try:
        while True:
            try:
                payload = json.loads(await ws.receive_text())
                msg_type = payload.get("type")

                if msg_type == "settings":
                    settings["server"] = payload.get("server", settings["server"])
                    settings["selected_server_id"] = payload.get("selected_server_id", settings["selected_server_id"])
                    settings["selected_model_id"] = payload.get("selected_model_id", settings["selected_model_id"])
                    tq = str(payload.get("target_quantization", settings["target_quantization"])).lower()
                    settings["target_quantization"] = tq if tq in {"none", "4bit", "8bit"} else "none"
                    dq = str(payload.get("draft_quantization", settings["draft_quantization"])).lower()
                    settings["draft_quantization"] = dq if dq in {"none", "4bit", "8bit"} else "none"
                    settings["cost"] = int(payload.get("cost", settings["cost"]))
                    settings["max_new_tokens"] = int(payload.get("max_new_tokens", settings["max_new_tokens"]))
                    obj_mode = str(payload.get("objective_selection_mode", settings["objective_selection_mode"])).lower()
                    settings["objective_selection_mode"] = "constraint" if obj_mode == "constraint" else "blend"
                    settings["metric_preference"] = str(
                        payload.get("metric_preference", settings["metric_preference"])
                    )
                    constraint_target = str(
                        payload.get("constraint_target", settings["constraint_target"])
                    ).lower()
                    settings["constraint_target"] = "tps" if constraint_target == "tps" else "metric"
                    settings["metric_constraint_per_1m_token"] = float(
                        payload.get("metric_constraint_per_1m_token", settings["metric_constraint_per_1m_token"])
                    )
                    settings["min_tps_constraint"] = float(
                        payload.get("min_tps_constraint", settings["min_tps_constraint"])
                    )
                    settings["algorithm"] = payload.get("algorithm", settings["algorithm"])
                    settings["draft_model_path"] = payload.get("draft_model_path", settings["draft_model_path"])
                    benchmark_dataset = str(payload.get("benchmark_dataset", settings["benchmark_dataset"])).lower()
                    if benchmark_dataset not in {"mt_bench", "gsm8k", "humaneval", "cnn_dailymail"}:
                        benchmark_dataset = "mt_bench"
                    settings["benchmark_dataset"] = benchmark_dataset
                    settings["proactive_drafting"] = payload.get("proactive_drafting", settings["proactive_drafting"])
                    if "online_profile_update" in payload:
                        settings["online_profile_update"] = bool(payload.get("online_profile_update"))
                    if "online_profile_lr" in payload:
                        try:
                            settings["online_profile_lr"] = float(payload.get("online_profile_lr"))
                        except Exception:
                            pass
                    mode = str(payload.get("mode", settings["mode"])).lower()
                    settings["mode"] = "chat" if mode == "chat" else "benchmark"
                    # Acknowledge updated runtime settings.
                    ok = await _safe_send_json(ws, {
                        "type": "settings_ack",
                        "server": settings["server"],
                        "selected_server_id": settings["selected_server_id"],
                        "selected_model_id": settings["selected_model_id"],
                        "target_quantization": settings["target_quantization"],
                        "draft_quantization": settings["draft_quantization"],
                        "cost": settings["cost"],
                        "max_new_tokens": settings["max_new_tokens"],
                        "objective_selection_mode": settings["objective_selection_mode"],
                        "metric_preference": settings["metric_preference"],
                        "constraint_target": settings["constraint_target"],
                        "metric_constraint_per_1m_token": settings["metric_constraint_per_1m_token"],
                        "min_tps_constraint": settings["min_tps_constraint"],
                        "algorithm": settings["algorithm"],
                        "draft_model_path": settings["draft_model_path"],
                        "benchmark_dataset": settings["benchmark_dataset"],
                        "proactive_drafting": settings["proactive_drafting"],
                        "online_profile_update": settings["online_profile_update"],
                        "online_profile_lr": settings["online_profile_lr"],
                        "mode": settings["mode"],
                    })
                    if not ok:
                        break
                    # Best-effort live update for persistent chat runtime.
                    await service.update_chat_runtime_settings(settings)
                    continue

                if msg_type == "control":
                    action = str(payload.get("action", "")).lower()
                    if action == "start":
                        if settings["mode"] == "benchmark":
                            msg = await service.start_run(settings)
                            if not await _safe_send_json(ws, {"type": "control_ack", "action": "start", "message": msg}):
                                break
                        else:
                            settings["chat_active"] = True
                            msg = await service.start_chat_session(settings)
                            if not await _safe_send_json(ws, {"type": "control_ack", "action": "start", "message": msg}):
                                break
                    elif action == "stop":
                        if settings["mode"] == "benchmark":
                            msg = await service.stop_run(settings)
                            if not await _safe_send_json(ws, {"type": "control_ack", "action": "stop", "message": msg}):
                                break
                        else:
                            settings["chat_active"] = False
                            msg = await service.stop_chat_session(settings)
                            if not await _safe_send_json(ws, {"type": "control_ack", "action": "stop", "message": msg}):
                                break
                    elif action == "shutdown_target":
                        settings["chat_active"] = False
                        msg = await service.shutdown_target(settings)
                        if not await _safe_send_json(ws, {"type": "control_ack", "action": "shutdown_target", "message": msg}):
                            break
                    continue

                if msg_type == "refresh_reference":
                    detailed_profile = bool(payload.get("detailed_profile", False))
                    if not await _safe_send_json(ws, {"type": "control_ack", "action": "refresh_reference", "message": "Refreshing reference cache..."}):
                        break
                    msg = await service.refresh_reference_cache(settings, detailed_profile=detailed_profile)
                    if not await _safe_send_json(ws, {"type": "control_ack", "action": "refresh_reference", "message": msg}):
                        break
                    continue

                if msg_type == "server_catalog_request":
                    if not await _safe_send_json(
                        ws,
                        {"type": "server_catalog", "servers": service.list_server_candidates()},
                    ):
                        break
                    continue

                if msg_type == "server_add":
                    result = await service.add_server_candidate(payload)
                    if not await _safe_send_json(ws, {"type": "server_add_result", **result}):
                        break
                    continue

                if msg_type == "server_update_key":
                    server_id = str(payload.get("server_id", ""))
                    api_key = str(payload.get("api_key", ""))
                    result = await service.update_server_api_key(server_id, api_key)
                    if not await _safe_send_json(ws, {"type": "server_update_key_result", **result}):
                        break
                    continue

                if msg_type == "server_remove":
                    server_id = str(payload.get("server_id", ""))
                    result = await service.remove_server_candidate(server_id)
                    if not await _safe_send_json(ws, {"type": "server_remove_result", **result}):
                        break
                    continue

                if msg_type == "probe_start":
                    if not await _safe_send_json(ws, {"type": "probe_status", "running": True}):
                        break
                    result = await service.run_probing(settings, payload)
                    if not await _safe_send_json(ws, {"type": "probe_result", **result}):
                        break
                    continue

                if msg_type == "recommendation_request":
                    metric_pref = str(payload.get("metric_preference", settings["metric_preference"]))
                    settings["metric_preference"] = metric_pref
                    result = await service.update_recommendations(settings, metric_pref)
                    if not await _safe_send_json(ws, {"type": "recommendation_result", **result}):
                        break
                    continue

                if msg_type == "chat":
                    user_text = (payload.get("message") or "").strip()
                    if not user_text:
                        continue

                    server = payload.get("server", settings["server"])
                    cost = int(payload.get("cost", settings["cost"]))
                    max_new_tokens = int(payload.get("max_new_tokens", settings["max_new_tokens"]))
                    obj_mode = str(payload.get("objective_selection_mode", settings["objective_selection_mode"])).lower()
                    objective_selection_mode = "constraint" if obj_mode == "constraint" else "blend"
                    constraint_target = str(
                        payload.get("constraint_target", settings["constraint_target"])
                    ).lower()
                    constraint_target = "tps" if constraint_target == "tps" else "metric"
                    metric_constraint_per_1m_token = float(
                        payload.get("metric_constraint_per_1m_token", settings["metric_constraint_per_1m_token"])
                    )
                    min_tps_constraint = float(
                        payload.get("min_tps_constraint", settings["min_tps_constraint"])
                    )
                    algorithm = payload.get("algorithm", settings["algorithm"])
                    benchmark_dataset = str(
                        payload.get("benchmark_dataset", settings["benchmark_dataset"])
                    ).lower()
                    if benchmark_dataset not in {"mt_bench", "gsm8k", "humaneval", "cnn_dailymail"}:
                        benchmark_dataset = settings["benchmark_dataset"]
                    proactive_drafting = payload.get("proactive_drafting", settings["proactive_drafting"])

                    # 1) " "
                    history.append({
                        "role": "user",
                        "content": user_text,
                        "server": server,
                        "cost": cost,
                        "max_new_tokens": max_new_tokens,
                        "objective_selection_mode": objective_selection_mode,
                        "constraint_target": constraint_target,
                        "metric_constraint_per_1m_token": metric_constraint_per_1m_token,
                        "min_tps_constraint": min_tps_constraint,
                        "algorithm": algorithm,
                        "benchmark_dataset": benchmark_dataset,
                        "proactive_drafting": proactive_drafting,
                        "mode": settings["mode"],
                    })
                    settings["max_new_tokens"] = max_new_tokens
                    settings["objective_selection_mode"] = objective_selection_mode
                    settings["constraint_target"] = constraint_target
                    settings["metric_constraint_per_1m_token"] = metric_constraint_per_1m_token
                    settings["min_tps_constraint"] = min_tps_constraint
                    settings["benchmark_dataset"] = benchmark_dataset

                    text = user_text.lower().strip()
                    token_trace = None
                    final_stats = None
                    if text in {"help", "/help"}:
                        reply = _help_text(settings["mode"])
                    elif settings["mode"] == "benchmark":
                        if text in {"status", "/status"}:
                            snap = service.snapshot()
                            reply = (
                                f"running={snap.get('running')}, exit_code={snap.get('exit_code')}\n"
                                f"throughput={snap.get('throughput', 0.0):.4f}, cost_per_1m={snap.get('cost_per_1m', 0.0):.4f}\n"
                                f"last_line={snap.get('last_line', '')}"
                            )
                        else:
                            # benchmark mode
                            reply = "Benchmark mode is active. Use Start/Stop for execution, and type `help` for guidance."
                    else:
                        if not settings.get("chat_active", True):
                            reply = "Chat session is stopped. Press Start to resume."
                        else:
                            if chat_task is not None and not chat_task.done():
                                ok = await _safe_send_json(ws, {
                                    "type": "error",
                                    "message": "A chat generation is already in progress.",
                                })
                                if not ok:
                                    break
                                continue

                            settings_snapshot = dict(settings)

                            async def _run_chat_turn(user_text_for_task: str, settings_for_task: dict):
                                try:
                                    async def _on_partial(partial_msg: dict):
                                        await _safe_send_json(ws, {
                                            "type": "chat_partial",
                                            "reply": str(partial_msg.get("reply", "")),
                                            "token_trace": partial_msg.get("token_trace", None),
                                            "stats": partial_msg.get("stats", None),
                                        })

                                    chat_result = await service.generate_chat_reply(
                                        user_text_for_task,
                                        settings_for_task,
                                        on_partial=_on_partial,
                                    )
                                    reply_text = str(chat_result.get("reply", ""))
                                    reply_trace = chat_result.get("token_trace", None)
                                    reply_final_stats = chat_result.get("final_stats", None)
                                    history.append({"role": "assistant", "content": reply_text})
                                    await _safe_send_json(ws, {
                                        "type": "chat_reply",
                                        "reply": reply_text,
                                        "token_trace": reply_trace,
                                        "final_stats": reply_final_stats if settings["mode"] == "chat" else None,
                                    })
                                    # Query trade-off/diag stats push.
                                    snap_now = service.snapshot()
                                    await _safe_send_json(ws, {
                                        "type": "stats",
                                        "gpu_energy": float(snap_now.get("gpu_energy", 0.0)),
                                        "draft_cost": float(snap_now.get("draft_cost", 0.0)),
                                        "target_cost": float(snap_now.get("target_cost", 0.0)),
                                        "throughput": float(snap_now.get("throughput", 0.0)),
                                        "running": bool(snap_now.get("running", False)),
                                        "chat_running": bool(snap_now.get("chat_running", False)),
                                        "chat_ready": bool(snap_now.get("chat_ready", False)),
                                        "chat_exit_code": snap_now.get("chat_exit_code"),
                                        "phase": str(snap_now.get("phase", "idle")),
                                        "phase_label": str(snap_now.get("phase_label", "Idle")),
                                        "last_line": str(snap_now.get("last_line", "")),
                                        "last_error": str(snap_now.get("last_error", "")),
                                        "feasible_constraint_range_per_1m": snap_now.get("feasible_constraint_range_per_1m"),
                                        "reference_tradeoff_curve_cs0_1": snap_now.get("reference_tradeoff_curve_cs0_1"),
                                        "reference_tradeoff_curve_by_constraint": snap_now.get("reference_tradeoff_curve_by_constraint"),
                                        "reference_cs_anchor_curve": snap_now.get("reference_cs_anchor_curve"),
                                        "reference_constraint_anchor_curve": snap_now.get("reference_constraint_anchor_curve"),
                                        "server_candidates": snap_now.get("server_candidates"),
                                        "probe_status": snap_now.get("probe_status"),
                                        "server_probe_rows": snap_now.get("server_probe_rows"),
                                        "server_probe_curves": snap_now.get("server_probe_curves"),
                                        "server_probe_curve_mode": snap_now.get("server_probe_curve_mode"),
                                        "recommendations": snap_now.get("recommendations"),
                                        "metric_preference": snap_now.get("metric_preference"),
                                        "benchmark_dataset": snap_now.get("benchmark_dataset"),
                                    })
                                except Exception as inner_e:
                                    await _safe_send_json(ws, {
                                        "type": "error",
                                        "message": str(inner_e),
                                    })

                            chat_task = asyncio.create_task(_run_chat_turn(user_text, settings_snapshot))
                            continue

                    history.append({"role": "assistant", "content": reply})

                    # 3) AI
                    ok = await _safe_send_json(ws, {
                        "type": "chat_reply",
                        "reply": reply,
                        "token_trace": token_trace,
                        "final_stats": final_stats if settings["mode"] == "chat" else None,
                    })
                    if not ok:
                        break
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                continue
            except Exception as e:
                print(f"Error processing message: {e}")
                if _is_ws_closed_error(e):
                    break
                ok = await _safe_send_json(ws, {"type": "error", "message": str(e)})
                if not ok:
                    break
                continue

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        stats_task.cancel()
        if chat_task is not None:
            chat_task.cancel()


async def send_stats_periodically(ws: WebSocket):
    """
    Periodically send stats data to clients.
    Use actual measurements in a real implementation.
    """
    try:
        while True:
            await asyncio.sleep(0.5)  # 0.5
            
            snap = service.snapshot()
            stats = {
                "type": "stats",
                "gpu_energy": float(snap.get("gpu_energy", 0.0)),
                "draft_cost": float(snap.get("draft_cost", 0.0)),
                "target_cost": float(snap.get("target_cost", 0.0)),
                "throughput": float(snap.get("throughput", 0.0)),
                "running": bool(snap.get("running", False)),
                "chat_running": bool(snap.get("chat_running", False)),
                "chat_ready": bool(snap.get("chat_ready", False)),
                "chat_exit_code": snap.get("chat_exit_code"),
                "phase": str(snap.get("phase", "idle")),
                "phase_label": str(snap.get("phase_label", "Idle")),
                "last_line": str(snap.get("last_line", "")),
                "last_error": str(snap.get("last_error", "")),
                "feasible_constraint_range_per_1m": snap.get("feasible_constraint_range_per_1m"),
                "reference_tradeoff_curve_cs0_1": snap.get("reference_tradeoff_curve_cs0_1"),
                "reference_tradeoff_curve_by_constraint": snap.get("reference_tradeoff_curve_by_constraint"),
                "reference_cs_anchor_curve": snap.get("reference_cs_anchor_curve"),
                "reference_constraint_anchor_curve": snap.get("reference_constraint_anchor_curve"),
                "server_candidates": snap.get("server_candidates"),
                "probe_status": snap.get("probe_status"),
                "server_probe_rows": snap.get("server_probe_rows"),
                "server_probe_curves": snap.get("server_probe_curves"),
                "server_probe_curve_mode": snap.get("server_probe_curve_mode"),
                "recommendations": snap.get("recommendations"),
                "metric_preference": snap.get("metric_preference"),
                "benchmark_dataset": snap.get("benchmark_dataset"),
            }
            
            ok = await _safe_send_json(ws, stats)
            if not ok:
                break
    except asyncio.CancelledError:
        pass

"""Plain dataclass that mirrors the ``Autodraft`` constructor arguments."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AutodraftConfig:
    draft_model: str
    target_model: str
    draft_quantization: Optional[str] = None
    target_quantization: Optional[str] = None
    # AutoDraft is always split-process — the draft side opens a socket to a
    # target server. ``target_host`` defaults to ``127.0.0.1`` so single-host
    # users (run target.py and draft.py in two terminals on the same machine)
    # don't have to repeat themselves; cross-host users override it.
    target_host: str = "127.0.0.1"
    target_port: int = 26001
    # Cost objective for the speculative decoding planner: which kind of
    # cost the runtime should minimize. One of ``"total_cost"``,
    # ``"api_cost"``, ``"draft_energy"``, ``"target_energy"``. Set at engine construction; ``run()`` no longer
    # exposes a per-call override (different cost objective ⇒ make a new
    # ``Autodraft`` instance, since the reference cache key depends on it).
    cost: str = "total_cost"
    # Hugging Face access token for gated repos (e.g. meta-llama/*). When
    # ``None``, ``run_local`` falls back to the ``HF_TOKEN`` /
    # ``HUGGING_FACE_HUB_TOKEN`` environment variables. Treat as a secret —
    # ``Autodraft.__repr__`` masks this field.
    hf_token: Optional[str] = None

"""Start the AutoDraft target server from a Python file.

Run this in one terminal first; it blocks forever (server loop). Then run
``examples/draft.py`` in another terminal.

The target runs in lazy-load mode: it has no model on the GPU until the
draft side connects and tells it which model to load. So you only set
``server_name`` (and optionally ``hf_token``) here — the actual model id,
quantization, and device mapping are decided by ``Autodraft(...)`` on the
draft side.
"""
from autodraft import serve_target  # noqa: E402

if __name__ == "__main__":
    serve_target(
        host="0.0.0.0",
        port=26001,
        # ``server_name`` must match what the draft side passes (default
        # ``"autodraft"`` on both ends). Different ``server_name`` →
        # separate profile/reference cache slot.
        server_name="autodraft",
        # Pass an HF token here for gated repos, or set HF_TOKEN in your
        # shell. Treat the token as a secret and do not commit it.
        hf_token=None,
        enable_auto_target_profile=True,
    )

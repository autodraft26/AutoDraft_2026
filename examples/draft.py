"""Run the draft side and produce generated text.

Symmetric counterpart to ``examples/target.py``: start that script in one
terminal first, then run this one in another. The two communicate over
``target_host:target_port`` (default ``127.0.0.1:26001``).
"""
from autodraft import Autodraft  # noqa: E402


if __name__ == "__main__":
    engine = Autodraft(
        draft_model="meta-llama/Llama-3.2-1B-Instruct",
        target_model="meta-llama/Llama-3.2-1B-Instruct",
        draft_quantization=None,
        target_quantization=None,
        target_host="127.0.0.1",
        target_port=26001,
        cost="total_cost",
        # Pass an HF token here for gated repos, or set HF_TOKEN in your
        # shell. Treat the token as a secret and do not commit it.
        hf_token=None,
    )

    result = engine.run(
        input_text="Write a short summary of speculative decoding.",
        proactive=False,
        cs="balanced",
        # Must match the server_name used on the target side.
        server_name="autodraft",
    )

    print("=" * 60)
    print("Generated text:")
    print("-" * 60)
    print(result["generated_text"])
    print("-" * 60)
    print(f"avg tree depth: {result['stats']['avg_tree_depth']:.2f}")
    print("=" * 60)
    print(f"avg nnodes: {result['stats']['avg_nnodes']:.2f}")
    print("=" * 60)
    print(f"accept length: {result['stats']['avg_accept_length']:.2f}")
    print("=" * 60)
    print(f"total cost: ${result['stats']['total_cost']:.6f}")
    print("=" * 60)
    print(f"trade-off saved: {result['tradeoff_files']}")
    print("=" * 60)
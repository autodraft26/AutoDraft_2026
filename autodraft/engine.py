"""Public ``Autodraft`` class.

Kept import-light: this module must not import torch / transformers / the
AutoDraft source tree. Heavy imports happen lazily inside
:mod:`autodraft.local_runner` when ``run()`` is actually invoked.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

from .config import AutodraftConfig
from .errors import InvalidAutodraftConfigError
# Pure-stdlib cost-metric validator. ``local_runner`` is import-light at
# module load (heavy imports happen inside ``_import_runtime``) so this
# import doesn't pull torch.
from .local_runner import _resolve_cost_metric

# Type aliases that show up in IDE hover so the valid options are visible
# inline on the function signature, not just buried in the docstring body.
CsPreset = Literal["tps", "balanced", "cost"]
CostMetric = Literal[
    "total_cost",
    "api_cost",
    "draft_energy",
    "target_energy",
]
QuantSpec = Optional[Literal["none", "4bit", "8bit"]]


class Autodraft:
    """Thin wrapper around the AutoDraft speculative-decoding runtime.

    AutoDraft always runs split-process: the draft side (this object) opens a
    socket to a target server (started via ``autodraft.serve_target`` or
    ``examples/target.py``). For single-host setups, leave ``target_host`` at
    its default ``"127.0.0.1"`` and run the target script in another
    terminal. For cross-host setups, set ``target_host`` to the target
    machine's address.

    Parameters
    ----------
    draft_model : str
        Hugging Face hub id (e.g. ``"meta-llama/Llama-3.2-1B-Instruct"``) or
        a local checkpoint path of the **small draft model**. The draft
        model proposes token trees; it should be much smaller than the
        target so that running it on every step is cheap.
        Must be non-empty.
    target_model : str
        Hugging Face hub id or local path of the **large target model** that
        verifies the draft trees and produces the final tokens. Must be
        non-empty. For real speed-up, ``target_model`` should be
        meaningfully larger than ``draft_model``; otherwise speculative
        decoding has nothing to gain.
    draft_quantization : {None, "none", "4bit", "8bit"}, optional
        Bitsandbytes quantization for the draft model. ``None`` and
        ``"none"`` are equivalent and mean fp16/bf16. ``"4bit"`` and
        ``"8bit"`` use bitsandbytes (a base dependency, no extra needed).
        Default ``None``.
    target_quantization : {None, "none", "4bit", "8bit"}, optional
        Same as ``draft_quantization`` but for the target model. Default
        ``None``.
    target_host : str, optional
        Hostname / IP of the AutoDraft target server. Default
        ``"127.0.0.1"`` covers the same-machine case (run
        ``examples/target.py`` in another terminal). For a target running
        on another machine, pass that machine's IP / hostname. Cannot be
        empty.
    target_port : int, optional
        TCP port the target server is listening on. Must match the
        ``port`` you passed to ``serve_target``. Default ``26001``.
    cost : {"total_cost", "api_cost", "draft_energy", "target_energy"}, optional
        Which cost metric the runtime should minimize. Set once at engine
        construction because the reference cache key depends on it
        (different ``cost`` ⇒ separate ``ref_*.json`` and ``tradeoff_*``
        files). To switch metrics, create a new ``Autodraft`` instance.

        - ``"total_cost"`` (**default**): combined draft+target dollar
          cost (GPU-seconds × per-second rate, plus communication).
        - ``"api_cost"``: target-side API/compute cost only.
        - ``"draft_energy"`` / ``"target_energy"``: per-side energy only.

        Energy metrics auto-enable GPU monitoring inside the runtime.
    hf_token : str, optional
        Hugging Face access token used for gated repositories
        (``meta-llama/*``, etc.). When ``None``, ``run()`` falls back to
        the ``HF_TOKEN`` and ``HUGGING_FACE_HUB_TOKEN`` environment
        variables. Treat the token as a secret — it is masked as ``'***'``
        in ``repr(engine)`` and never printed. Default ``None``.
    **kwargs
        Any extra option forwarded verbatim to the underlying
        ``run_draft``. Common pass-throughs:

        - ``server_name`` (str): must match the target's ``server_name``
          for profile/reference cache reuse. Default ``"autodraft"`` on
          both sides.
        - ``nodes`` (int): max nodes per tree. Default ``150``.
        - ``max_depth`` (int): max tree depth. Default ``15``.
        - ``algorithm`` (str): ``"AutoDraft"`` (default), ``"Server-Only"``,
          ``"Server-Only-AR"``, ``"OPT-Tree"``, or ``"Fixed-tree"``. Can
          also be passed at ``run()`` time.

        Anything ``run_draft`` doesn't recognize is dropped with a log
        line, so unknown keys are safe.

    Raises
    ------
    InvalidAutodraftConfigError
        On empty ``draft_model`` / ``target_model`` / ``target_host``,
        non-int ``target_port``, or non-string ``hf_token``.
    """

    def __init__(
        self,
        draft_model: str,
        target_model: str,
        draft_quantization: QuantSpec = None,
        target_quantization: QuantSpec = None,
        target_host: str = "127.0.0.1",
        target_port: int = 26001,
        cost: CostMetric = "total_cost",
        hf_token: Optional[str] = None,
        **kwargs: Any,
    ):
        if not isinstance(draft_model, str) or not draft_model:
            raise InvalidAutodraftConfigError("draft_model must be a non-empty string")
        if not isinstance(target_model, str) or not target_model:
            raise InvalidAutodraftConfigError("target_model must be a non-empty string")
        if not isinstance(target_host, str) or not target_host:
            raise InvalidAutodraftConfigError("target_host must be a non-empty string")
        if not isinstance(target_port, int) or isinstance(target_port, bool):
            raise InvalidAutodraftConfigError("target_port must be int")
        if hf_token is not None and not isinstance(hf_token, str):
            raise InvalidAutodraftConfigError("hf_token must be a string or None")
        try:
            cost_metric = _resolve_cost_metric(cost)
        except ValueError as exc:
            raise InvalidAutodraftConfigError(str(exc)) from exc

        self.config = AutodraftConfig(
            draft_model=draft_model,
            target_model=target_model,
            draft_quantization=draft_quantization,
            target_quantization=target_quantization,
            target_host=target_host,
            target_port=target_port,
            cost=cost_metric,
            hf_token=hf_token,
        )
        self.extra_kwargs = dict(kwargs)

    def run(
        self,
        input_text: str,
        proactive: bool = False,
        cs: Union[CsPreset, float] = "balanced",
        save_tradeoff: bool = True,
        tradeoff_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Run speculative decoding on ``input_text`` and return the result.

        The target server (``serve_target`` / ``examples/target.py``) must
        already be running and reachable at the ``target_host:target_port``
        configured on this ``Autodraft`` instance.

        Parameters
        ----------
        input_text : str
            The user prompt to feed into the model. Cannot be empty.
            One prompt per ``run()`` call — wrap a chat history into a
            single string yourself if you need multi-turn context.
        proactive : bool, optional
            Enable **proactive drafting**: speculatively start the next
            draft tree while the previous one is still being verified by
            the target, reducing tail latency. Wires up
            ``--proactive-drafting`` and ``--adaptive-proactive-threshold``
            in the underlying runtime. Default ``False``.
        cs : {"tps", "balanced", "cost"} or float, optional
            Cost-sensitivity. Controls the trade-off between throughput
            (TPS) and per-token cost when the runtime picks tree shapes.

            - ``"tps"`` (== 0): maximize tokens per second, ignore cost.
            - ``"balanced"`` (== 0.5, **default**): equal blend.
            - ``"cost"`` (== 1): minimize cost, accept lower TPS.
            - any number in ``[0, 1]``: linear interpolation for fine
              control.

            Numbers outside ``[0, 1]`` and unrecognized strings raise
            ``ValueError``. The interpretation of "cost" is set by the
            ``cost`` parameter on ``Autodraft(...)`` (e.g. dollars vs. kWh).
        save_tradeoff : bool, optional
            When ``True`` (**default**), the reference cs-0~1 trade-off
            curve is dumped to disk every run. The curve plots predicted
            TPS vs. predicted cost-per-1M-tokens across the cs sweep —
            same data the chat UI graphs in real time. JSON is always
            written; PNG is rendered via matplotlib (a base dep).
            Default ``True``.

            **Filename is conditions-hashed**, so repeated calls with the
            same ``(server_name, target_model, draft_model, device,
            quantization, bench_name, cost, objective_mode)`` overwrite
            the **same** file rather than piling up timestamped copies.
            The on-disk artifact therefore always reflects the latest
            reference state for those conditions. Different conditions
            land in different files (paired 1:1 with the reference cache
            naming).
        tradeoff_dir : str, optional
            Override directory for trade-off artifacts. When ``None``
            (default), files land in ``$AUTODRAFT_DATA_DIR/tradeoff`` so
            they sit next to the profile/reference caches.
        **kwargs
            Forwarded to ``run_draft``. The wrapper merges these on top of
            any kwargs you passed to ``Autodraft(...)``. Common ones:

            - ``algorithm`` (str): AutoDraft variant. ``"AutoDraft"``
              (default) / ``"Server-Only"`` / ``"Server-Only-AR"`` /
              ``"OPT-Tree"`` / ``"Fixed-tree"``.
            - ``server_name`` (str): must match what was passed on the
              target side; default ``"autodraft"`` on both ends.
            - ``nodes`` (int), ``max_depth`` (int): tree size knobs.
            - ``seed`` (int): random seed; default ``4``.

            Unknown keys are dropped with a console log line.

        Returns
        -------
        dict
            ``{generated_text, input_text, proactive, cs, cost, algorithm,
            stats, tradeoff_files, answer_row, raw_result}``.

            - ``generated_text`` (str): final model output.
            - ``stats`` (dict): one-line summary — ``total_steps``,
              ``total_new_tokens``, ``total_time_seconds``,
              ``tokens_per_second``, ``tokens_per_step``,
              ``avg_tree_width``, ``avg_tree_depth``, ``avg_nnodes``,
              ``avg_accept_length``, ``acceptance_ratio_avg``,
              ``total_cost``, ``api_cost``, ``draft_cost``,
              ``target_cost``.
            - ``tradeoff_files`` (dict or None): paths of any artifacts
              that got written (``{"json": ..., "png": ...}``).
            - ``raw_result`` (dict): the full integrated result file the
              runtime writes — has detailed ``latency_statistics``,
              ``accept_stats``, ``draft_gpu_summary``, etc.

        Raises
        ------
        ValueError
            On empty ``input_text``, invalid ``cs`` (unknown string or
            out-of-range number), or invalid ``cost`` metric name.
        autodraft.errors.LocalRuntimeUnavailableError
            If the heavy AutoDraft runtime is not importable in this
            environment (e.g. PyTorch / fschat missing).
        autodraft.errors.RemoteTargetConnectionError
            (Surfaced by ``run_draft``'s socket layer.) When the target
            server isn't reachable at ``target_host:target_port``.
        """

        if not isinstance(input_text, str) or not input_text:
            raise ValueError("input_text must be a non-empty string")

        merged_kwargs = dict(self.extra_kwargs)
        merged_kwargs.update(kwargs)

        from .local_runner import run_local

        return run_local(
            draft_model=self.config.draft_model,
            target_model=self.config.target_model,
            draft_quantization=self.config.draft_quantization,
            target_quantization=self.config.target_quantization,
            input_text=input_text,
            proactive=proactive,
            cs=cs,
            cost=self.config.cost,
            target_host=self.config.target_host,
            target_port=self.config.target_port,
            hf_token=self.config.hf_token,
            save_tradeoff=save_tradeoff,
            tradeoff_dir=tradeoff_dir,
            **merged_kwargs,
        )

    def __repr__(self) -> str:
        c = self.config
        token_repr = "None" if c.hf_token is None else "'***'"
        return (
            f"Autodraft(draft_model={c.draft_model!r}, "
            f"target_model={c.target_model!r}, "
            f"target_host={c.target_host!r}, "
            f"target_port={c.target_port}, "
            f"cost={c.cost!r}, "
            f"hf_token={token_repr})"
        )

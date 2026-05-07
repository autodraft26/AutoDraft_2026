import torch
import random
import time
import json
import math
import bisect


def _normalize_objective_metric(value):
    metric = str(value).lower() if value is not None else "total_cost"
    if metric == "cost":
        return "total_cost"
    return metric


class Tree:
    def __init__(
        self,
        nnodes,
        device,
        max_depth,
        per_token_probability_bound=0.0,
        per_path_probability_bound=0.0,
        fixed_width=None,
        fixed_nnodes=False,
        fixed_depth=False,
        min_width=3,
        profile_data=None,
        target_profile_data=None,
        draft_per_sec_cost=0.0,
        target_per_sec_cost=0.0,
        cost_sensitivity=0.0,
        reference_tps: float = 1.0,
        reference_objective_per_token: float = 1.0,
        objective_metric: str = "total_cost",
        per_token_draft_to_target_transfer_time=0.0,
        per_token_target_to_draft_transfer_time=0.0,
        per_token_draft_to_target_bytes=0.0,
        per_token_target_to_draft_bytes=0.0,
        user_communication_cost_per_gb=0.09,
        cloud_inbound_cost_per_gb=None,
        cloud_outbound_cost_per_gb=0.09,
        communication_per_gb_cost=None,
        target_time_scale: float = 1.0,
        accept_length_scale: float = 1.0,
        accept_length_margin: float = 0.05,
        objective_selection_mode: str = "blend",
        constraint_target: str = "metric",
        metric_constraint_per_token: float = None,
        min_tps_constraint: float = None,
        opt_tree: bool = False,
        no_draft_cost: bool = False,
        stop_flag=None,
        proactive_time_budget_sec: float = None,
        proactive_continue_event=None,
        proactive_use_probability: float = None,
        proactive_depth_stats: dict = None,
        proactive_disable_budget: bool = False,
    ):
        self.max_nnodes = nnodes  # (max_width)
        self.device = device
        self.depth = 0
        self.weight = 0
        self.per_token_probability_bound = per_token_probability_bound
        self.per_path_probability_bound = per_path_probability_bound
        self.max_depth = max_depth
        self.fixed_width = fixed_width  # : width (None )
        self.fixed_nnodes = fixed_nnodes
        self.fixed_depth = fixed_depth
        self.min_width = min_width
        # (width model_call_avg_time_ms )
        self.profile_data = profile_data  # {width: {"model_call_avg_time_ms": ...}, ...}
        # Target (width_depth_nnodes avg_time_ms )
        self.target_profile_data = target_profile_data  # {"nnodes_Z": {"max_nnodes": Z, "avg_time_ms": ...}, ...}
        # Target ( / , )
        self.target_time_scale = float(target_time_scale) if target_time_scale is not None else 1.0
        # expected_accept_length ( / , )
        self.accept_length_scale = float(accept_length_scale) if accept_length_scale is not None else 1.0
        # expected accept length (5%):
        self.accept_length_margin = max(0.0, min(0.99, float(accept_length_margin)))
        # tree 
        self.prev_final_nnodes = 0  # tree top_index (final_width)
        self.prev_depth = 0  # tree depth
        self.prev_width = 0  # tree width
        # (DraftRunner )
        self.per_token_draft_to_target_transfer_time = per_token_draft_to_target_transfer_time  # Draft Target 
        self.per_token_target_to_draft_transfer_time = per_token_target_to_draft_transfer_time  # Target Draft 
        self.per_token_draft_to_target_bytes = float(per_token_draft_to_target_bytes or 0.0)
        self.per_token_target_to_draft_bytes = float(per_token_target_to_draft_bytes or 0.0)
        resolved_user_cost = user_communication_cost_per_gb
        if resolved_user_cost is None:
            if cloud_inbound_cost_per_gb is not None:
                resolved_user_cost = cloud_inbound_cost_per_gb
            elif communication_per_gb_cost is not None:
                resolved_user_cost = communication_per_gb_cost
            else:
                resolved_user_cost = 0.09
        self.user_communication_cost_per_gb = max(0.0, float(resolved_user_cost or 0.0))
        self.cloud_outbound_cost_per_gb = max(0.0, float(cloud_outbound_cost_per_gb or 0.0))
        self.draft_per_sec_cost = draft_per_sec_cost
        self.target_per_sec_cost = target_per_sec_cost
        # Cost sensitivity weight in [0, 1].
        self.cost_sensitivity = cost_sensitivity
        # Tree candidate selection mode: blend | constraint
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
        self.reference_tps = float(reference_tps) if reference_tps is not None else 1.0
        self.reference_objective_per_token = (
            float(reference_objective_per_token)
            if reference_objective_per_token is not None else 1.0
        )
        # Objective metric:
        # - cost
        # - draft_energy
        # - target_energy
        self.objective_metric = _normalize_objective_metric(objective_metric)
        # draft tree (latency )
        self.no_draft_cost = bool(no_draft_cost)
        # Depth (JSON )
        self.depth_stats = {}  # {depth: {"prev_sum_expected_accepted_length": [...], "sum_expected_accepted_length": [...], "per_token_latency": [...], "per_token_cost": [...]}}
        # objective_value (depth )
        self.prev_objective_value = None
        # opt-tree (weight mode)
        self.opt_tree = bool(opt_tree)
        self.width_algorithm_times = []  # width 
        self.nnodes_algorithm_times = []  # final_nnodes 
        self.weight_matrix = torch.zeros([max_depth, self.max_nnodes], device=self.device)
        self.input_ids_matrix = torch.zeros([max_depth, self.max_nnodes], dtype=torch.long, device=self.device)
        self.parents_matrix = torch.zeros([max_depth, self.max_nnodes], dtype=torch.long, device=self.device)
        # depth width
        self.depth_widths = []
        # depth width 
        self.current_width = 0
        # width model.model()
        self.width_times = {}  # {width: [time1, time2, ...]}
        self._width_time_running_sum = {}  # {width: sum_time}
        self._width_time_running_count = {}  # {width: count}
        self._width_pred_time_cache_version = 0
        self._width_pred_time_cache = {}  # {(version, device, widths_tuple): torch.Tensor}
        # proactive budget
        self.last_observed_draft_time_sec = 0.0
        self.last_observed_width = 0
        # (probability <= 5% )
        self.proactive_prediction_target_quantile = 0.95
        # fallback (warmup)
        self.proactive_prediction_warmup_guard_ratio = 1.50
        self.proactive_prediction_min_samples = 32
        self.proactive_prediction_max_samples = 256
        self._proactive_prediction_error_ratios = []  # global rolling window
        self._proactive_prediction_error_ratios_by_width = {}  # {width: [ratio, ...]}
        # tree draft
        self.draft_total_time = 0.0
        # draft
        self.prev_draft_total_time = 0.0
        # tree draft ( draft_time )
        self.expected_draft_total_time = 0.0
        # accept length
        self.sum_expected_accepted_length = 1.0
        # accept length
        self.prev_sum_expected_accepted_length = 1.0
        self.prev_target_time = 0.2  # target 
        self.prev_per_token_latency = None  # per_token_latency
        self.prev_per_token_cost = None  # per_token_cost
        # breakpoint()
        self.final_nnodes = 0
        # proactive draft (threading.Event)
        self.stop_flag = stop_flag
        # proactive budget (target verification )
        self.proactive_time_budget_sec = (
            float(proactive_time_budget_sec)
            if proactive_time_budget_sec is not None and float(proactive_time_budget_sec) > 0
            else None
        )
        self.proactive_continue_event = proactive_continue_event
        self.proactive_disable_budget = bool(proactive_disable_budget)
        self.proactive_budget_wait_sec = 0.0
        self.proactive_use_probability = (
            max(0.0, min(1.0, float(proactive_use_probability)))
            if proactive_use_probability is not None
            else None
        )
        self.proactive_depth_stats = proactive_depth_stats if isinstance(proactive_depth_stats, dict) else None
        self.proactive_expand_continue_count = 0
        self.proactive_expand_pause_count = 0
        self.proactive_finalize_early_count = 0
        self.proactive_expected_gain_sec = 0.0
        self.proactive_expected_loss_sec = 0.0
        self.proactive_last_expand_decision = None
        self.proactive_expand_depth_counts = {}
        self.proactive_expected_gain_by_depth = {}
        self.proactive_expected_loss_by_depth = {}
        # target profile lookup
        self.target_profile_lookup_stats = {
            "direct_hit": 0,
            "interpolated_hit": 0,
            "nearest_hit": 0,
            "fallback": 0,
        }
        # constraint (fallback )
        self.constraint_decision_stats = {
            "width_candidate_feasible": 0,
            "width_candidate_infeasible": 0,
            "width_selected_feasible": 0,
            "width_selected_fallback": 0,
            "nnodes_candidate_feasible": 0,
            "nnodes_candidate_infeasible": 0,
            "nnodes_selected_feasible": 0,
            "nnodes_selected_fallback": 0,
        }
        # target profile (nnodes -> avg_time_ms)
        self._target_profile_points = []
        self._target_profile_nnodes = []
        self._rebuild_target_profile_cache()
        self._target_lookup_runtime_cache = {}
        # update() finalize 
        self.last_finalize_time_sec = 0.0

    def _rebuild_target_profile_cache(self):
        """Preprocess the target profile dict once with alignment points for interpolation."""
        points_by_nnodes = {}
        if isinstance(self.target_profile_data, dict):
            for _, v in self.target_profile_data.items():
                try:
                    n = int(v.get("max_nnodes"))
                    t = float(v.get("avg_time_ms", 0.0))
                except Exception:
                    continue
                if n <= 0:
                    continue
                if n not in points_by_nnodes:
                    points_by_nnodes[n] = t
        sorted_points = sorted(points_by_nnodes.items(), key=lambda x: x[0])
        self._target_profile_points = sorted_points
        self._target_profile_nnodes = [n for n, _ in sorted_points]
        self._target_lookup_runtime_cache = {}

    def _invalidate_width_prediction_cache(self):
        self._width_pred_time_cache_version += 1
        self._width_pred_time_cache = {}

    def _predict_next_time_for_width(self, width: int) -> float:
        width_i = int(width)
        if self.profile_data is not None:
            row = self.profile_data.get(str(width_i))
            if isinstance(row, dict):
                return float(row.get("model_call_avg_time_ms", 0.0)) / 1000.0
        cnt = int(self._width_time_running_count.get(width_i, 0))
        if cnt > 0:
            return float(self._width_time_running_sum.get(width_i, 0.0)) / float(cnt)
        return 0.01

    def _predict_next_time_for_width_conservative(self, width: int) -> float:
        """
        proactive budget gating conservative prediction for.
        - profile/running-average prediction
        - prediction based on recent measurements (draft_time) with width-ratio correction
        Use the larger value and multiply it by the upper quantile of prediction error (default 95%).
        """
        width_i = int(max(1, width))
        candidates = []

        base_pred = float(self._predict_next_time_for_width(width_i))
        if base_pred > 0:
            candidates.append(base_pred)

        last_obs = float(self.last_observed_draft_time_sec or 0.0)
        last_w = int(max(1, self.last_observed_width or 1))
        if last_obs > 0:
            width_scale = float(width_i) / float(last_w)
            width_scale = max(0.5, min(2.5, width_scale))
            candidates.append(last_obs * width_scale)

        if not candidates:
            candidates = [0.01]

        guarded = max(candidates) * float(self._get_proactive_prediction_guard_ratio(width_i))
        return max(0.0, float(guarded))

    def _record_proactive_prediction_error_ratio(self, width: int, predicted_sec: float, observed_sec: float):
        """Next expansion time prediction error multiplier r=observed/predicted is recorded in the rolling window."""
        try:
            pred = float(predicted_sec)
            obs = float(observed_sec)
        except Exception:
            return
        if pred <= 0 or obs <= 0:
            return
        ratio = obs / pred
        if not math.isfinite(ratio):
            return
        # outlier 
        ratio = max(0.25, min(8.0, float(ratio)))

        self._proactive_prediction_error_ratios.append(ratio)
        if len(self._proactive_prediction_error_ratios) > int(self.proactive_prediction_max_samples):
            del self._proactive_prediction_error_ratios[0]

        w = int(max(1, width))
        bucket = self._proactive_prediction_error_ratios_by_width.setdefault(w, [])
        bucket.append(ratio)
        if len(bucket) > int(self.proactive_prediction_max_samples):
            del bucket[0]

    def _get_proactive_prediction_guard_ratio(self, width: int) -> float:
        """
        Use the quantile of the error multiplier distribution (default 95%) as a guard.
        - Prefer the width bucket when enough samples exist for that width
        - Use the global bucket when samples are insufficient
        - Use the warmup fallback when both are insufficient
        """
        min_n = int(self.proactive_prediction_min_samples)
        q = float(self.proactive_prediction_target_quantile)
        q = max(0.5, min(0.999, q))
        warmup = float(self.proactive_prediction_warmup_guard_ratio)

        width_bucket = self._proactive_prediction_error_ratios_by_width.get(int(max(1, width)), [])
        if len(width_bucket) >= min_n:
            samples = width_bucket
        elif len(self._proactive_prediction_error_ratios) >= min_n:
            samples = self._proactive_prediction_error_ratios
        else:
            return max(1.0, warmup)

        sorted_samples = sorted(samples)
        idx = int(math.ceil(q * len(sorted_samples))) - 1
        idx = max(0, min(idx, len(sorted_samples) - 1))
        ratio = float(sorted_samples[idx])
        # budget guard 1.0
        return max(1.0, ratio)

    def _get_proactive_depth_survival_factor(self, depth: int) -> float:
        """Correct p_use with the used/canceled ratio for each observed depth."""
        if not isinstance(self.proactive_depth_stats, dict):
            return 1.0
        rec = self.proactive_depth_stats.get(int(depth))
        if not isinstance(rec, dict):
            return 1.0
        used = int(rec.get("used", 0) or 0)
        canceled = int(rec.get("canceled", 0) or 0)
        total = used + canceled
        if total < 20:
            return 1.0
        return max(0.05, min(1.0, float(used) / float(total)))

    def _record_proactive_expand_decision(self, decision: str, depth: int, gain_sec: float, loss_sec: float):
        self.proactive_last_expand_decision = str(decision)
        if decision == "continue":
            self.proactive_expand_continue_count += 1
        elif decision == "pause":
            self.proactive_expand_pause_count += 1
        elif decision == "finalize_early":
            self.proactive_finalize_early_count += 1
        depth_i = int(depth)
        self.proactive_expand_depth_counts[depth_i] = self.proactive_expand_depth_counts.get(depth_i, 0) + 1
        self.proactive_expected_gain_sec += max(0.0, float(gain_sec))
        self.proactive_expected_loss_sec += max(0.0, float(loss_sec))
        self.proactive_expected_gain_by_depth[depth_i] = (
            self.proactive_expected_gain_by_depth.get(depth_i, 0.0) + max(0.0, float(gain_sec))
        )
        self.proactive_expected_loss_by_depth[depth_i] = (
            self.proactive_expected_loss_by_depth.get(depth_i, 0.0) + max(0.0, float(loss_sec))
        )

    def _evaluate_proactive_expand_value(
        self,
        predicted_elapsed_sec: float,
        predicted_next_sec: float,
        budget_sec: float,
    ):
        """
        proactive expansion marginal value.
        Returns: (decision, expected_gain_sec, expected_loss_sec)
        """
        base_p = self.proactive_use_probability
        if base_p is None:
            return "continue", 0.0, 0.0
        p_use = max(0.0, min(1.0, float(base_p) * self._get_proactive_depth_survival_factor(self.depth)))
        next_sec = max(0.0, float(predicted_next_sec))
        elapsed_sec = max(0.0, float(predicted_elapsed_sec))
        budget = max(0.0, float(budget_sec))
        remain_sec = max(0.0, budget - elapsed_sec)
        hidden_sec = min(next_sec, remain_sec)
        exposed_sec = max(0.0, next_sec - remain_sec)

        # hidden main path draft build .
        expected_gain_sec = p_use * hidden_sec
        wasted_compute_sec = (1.0 - p_use) * next_sec
        expected_cancel_tail_sec = (1.0 - p_use) * min(next_sec, 0.050)
        exposed_wait_sec = p_use * exposed_sec
        expected_loss_sec = wasted_compute_sec + expected_cancel_tail_sec + exposed_wait_sec

        alpha = max(0.0, min(1.0, float(self.cost_sensitivity)))
        latency_norm = max(1e-9, 1.0 / max(1e-9, float(self.reference_tps)))
        metric_norm = max(1e-12, float(self.reference_objective_per_token))
        metric_loss = wasted_compute_sec * self._draft_objective_rate()
        value = (
            (1.0 - alpha) * ((expected_gain_sec - exposed_wait_sec) / latency_norm)
            - alpha * (metric_loss / metric_norm)
            - (expected_cancel_tail_sec / latency_norm)
        )

        if value < 0.0:
            # reply tree .
            # draft tree objective .
            return "pause", expected_gain_sec, expected_loss_sec
        if exposed_sec > 0.0:
            return "pause", expected_gain_sec, expected_loss_sec
        return "continue", expected_gain_sec, expected_loss_sec

    def _get_width_predicted_times_tensor(self, candidate_widths, device):
        widths_tuple = tuple(int(w) for w in candidate_widths)
        device_key = f"{device.type}:{device.index if device.index is not None else -1}"
        cache_key = (self._width_pred_time_cache_version, device_key, widths_tuple)
        cached = self._width_pred_time_cache.get(cache_key)
        if cached is not None:
            return cached
        vals = [self._predict_next_time_for_width(w) for w in widths_tuple]
        out = torch.tensor(vals, dtype=torch.float32, device=device)
        self._width_pred_time_cache[cache_key] = out
        return out

    def _lookup_target_time_cached(self, nnodes: int, default_time: float = 0.2) -> float:
        key = (int(nnodes), round(float(default_time), 9), round(float(self.target_time_scale), 9))
        hit = self._target_lookup_runtime_cache.get(key)
        if hit is not None:
            return hit
        val = self._lookup_target_time(nnodes=int(nnodes), default_time=float(default_time))
        self._target_lookup_runtime_cache[key] = float(val)
        return float(val)

    def _should_sync_timing(self, value) -> bool:
        """The update() internal add timing is always synchronized if it is a CUDA input."""
        is_cuda_tensor = (
            (isinstance(value, torch.Tensor) and value.is_cuda)
            or (
                isinstance(value, (list, tuple))
                and len(value) > 0
                and isinstance(value[0], torch.Tensor)
                and value[0].is_cuda
            )
        )
        if not is_cuda_tensor:
            return False
        return True

    def _uses_total_cost_objective(self) -> bool:
        return self.objective_metric == "total_cost"

    def _uses_api_cost_objective(self) -> bool:
        return self.objective_metric == "api_cost"

    def _uses_any_cost_objective(self) -> bool:
        return self.objective_metric in {"total_cost", "api_cost"}

    def _uses_draft_objective(self) -> bool:
        return self.objective_metric in {"total_cost", "draft_energy", "_combined_energy"}

    def _uses_target_objective(self) -> bool:
        return self.objective_metric in {"total_cost", "api_cost", "target_energy", "_combined_energy"}

    def _draft_objective_rate(self) -> float:
        """Draft objective rate per second (cost or energy)."""
        if self.no_draft_cost or not self._uses_draft_objective():
            return 0.0
        return float(self.draft_per_sec_cost)

    def _transfer_objective_rate(self) -> float:
        """Transfer objective rate per second (total_cost only)."""
        return float(self.draft_per_sec_cost) if self._uses_total_cost_objective() else 0.0

    def _target_objective_rate(self) -> float:
        """Target objective rate per second (cost or target-side energy)."""
        return float(self.target_per_sec_cost) if self._uses_target_objective() else 0.0

    def _transfer_objective_cost_from_tokens(self, d2t_tokens: float, t2d_tokens: float) -> float:
        """Objective cost based on communication charges."""
        if not self._uses_any_cost_objective():
            return 0.0

        bytes_per_gb = float(1024 ** 3)
        if bytes_per_gb <= 0:
            return 0.0

        d2t_bytes = float(d2t_tokens) * self.per_token_draft_to_target_bytes
        t2d_bytes = float(t2d_tokens) * self.per_token_target_to_draft_bytes
        inbound_cost = (d2t_bytes / bytes_per_gb) * self.user_communication_cost_per_gb
        user_outbound_cost = (t2d_bytes / bytes_per_gb) * self.user_communication_cost_per_gb
        cloud_outbound_cost = (t2d_bytes / bytes_per_gb) * self.cloud_outbound_cost_per_gb

        if self._uses_total_cost_objective():
            return max(0.0, inbound_cost + user_outbound_cost + cloud_outbound_cost)
        if self._uses_api_cost_objective():
            return max(0.0, user_outbound_cost + cloud_outbound_cost)
        return 0.0

    def _lookup_target_time(self, nnodes: int, default_time: float = 0.2) -> float:
        """Check nnodes standard time in target profile. In case of a miss, linear interpolation is performed, and the nearest value is used for areas outside the range."""
        if self.target_profile_data is None:
            self.target_profile_lookup_stats["fallback"] += 1
            return float(default_time)

        key = f"nnodes_{int(nnodes)}"
        direct = self.target_profile_data.get(key)
        if direct is not None:
            self.target_profile_lookup_stats["direct_hit"] += 1
            return (float(direct.get("avg_time_ms", 0.0)) * self.target_time_scale) / 1000.0

        sorted_points = self._target_profile_points
        sorted_nnodes = self._target_profile_nnodes
        if sorted_points:
            query_nnodes = int(nnodes)
            idx = bisect.bisect_left(sorted_nnodes, query_nnodes)
            if idx < len(sorted_nnodes) and sorted_nnodes[idx] == query_nnodes:
                self.target_profile_lookup_stats["nearest_hit"] += 1
                return (float(sorted_points[idx][1]) * self.target_time_scale) / 1000.0

            lower = sorted_points[idx - 1] if idx > 0 else None
            upper = sorted_points[idx] if idx < len(sorted_points) else None

            if lower is not None and upper is not None and upper[0] > lower[0]:
                ratio = (query_nnodes - lower[0]) / float(upper[0] - lower[0])
                interp_time_ms = float(lower[1]) + ratio * (float(upper[1]) - float(lower[1]))
                self.target_profile_lookup_stats["interpolated_hit"] += 1
                return (interp_time_ms * self.target_time_scale) / 1000.0

            nearest_time_ms = float(sorted_points[0][1]) if idx <= 0 else float(sorted_points[-1][1])
            self.target_profile_lookup_stats["nearest_hit"] += 1
            return (nearest_time_ms * self.target_time_scale) / 1000.0

        self.target_profile_lookup_stats["fallback"] += 1
        return float(default_time)

    def _sensitivity_alpha(self) -> float:
        """Return the cost sensitivity weight clamped to [0, 1]."""
        try:
            alpha = float(self.cost_sensitivity)
        except Exception:
            alpha = 0.0
        return max(0.0, min(1.0, alpha))

    def _apply_accept_conservative_margin(self, value: float) -> float:
        """Apply a payoff margin to the expected accept length denominator."""
        v = float(value)
        return v * max(0.0, 1.0 - float(self.accept_length_margin))

    def _normalized_blended_objective(self, total_latency: float, total_objective_cost: float, denom_sum: float) -> float:
        """
        objective = alpha * normalized_cost + (1-alpha) * normalized_tps
        - normalized_cost = objective_per_token / reference_objective_per_token
        - normalized_tps = reference_tps / current_tps (minimization form)
        """
        if denom_sum <= 0 or total_latency <= 0:
            return float("inf")
        current_tps = float(denom_sum) / float(total_latency)
        if current_tps <= 0:
            return float("inf")
        current_objective_per_token = float(total_objective_cost) / float(denom_sum)
        ref_tps = max(1e-9, float(self.reference_tps))
        ref_obj = max(1e-12, float(self.reference_objective_per_token))
        normalized_tps = ref_tps / current_tps
        normalized_cost = current_objective_per_token / ref_obj
        alpha = self._sensitivity_alpha()
        return (alpha * normalized_cost) + ((1.0 - alpha) * normalized_tps)

    def _constraint_objective(
        self,
        total_latency: float,
        total_objective_cost: float,
        denom_sum: float,
        stage: str = None,
    ) -> float:
        """
        Constraint mode objective:
        - constraint_target == "metric":
          metric_per_token <= metric_constraint_per_token among candidates where
          minimize per_token_latency (=1/tps)
          if no feasible candidate exists, minimize metric_per_token
        - constraint_target == "tps":
          current_tps >= min_tps_constraint among candidates where
          minimize metric_per_token
          if no feasible candidate exists, minimize per_token_latency (=1/tps)
        """
        if denom_sum <= 0 or total_latency <= 0:
            return float("inf")
        metric_per_token = float(total_objective_cost) / float(denom_sum)
        per_token_latency = float(total_latency) / float(denom_sum)
        current_tps = 1.0 / max(1e-9, per_token_latency)

        if self.constraint_target == "tps":
            feasible = not (
                self.min_tps_constraint is not None
                and self.min_tps_constraint > 0
                and current_tps < float(self.min_tps_constraint)
            )
            feasible_objective = metric_per_token
            fallback_objective = per_token_latency
        else:
            feasible = not (
                self.metric_constraint_per_token is not None
                and self.metric_constraint_per_token > 0
                and metric_per_token > float(self.metric_constraint_per_token)
            )
            feasible_objective = per_token_latency
            fallback_objective = metric_per_token

        if feasible:
            if stage == "width":
                self.constraint_decision_stats["width_candidate_feasible"] += 1
            elif stage == "nnodes":
                self.constraint_decision_stats["nnodes_candidate_feasible"] += 1
            return feasible_objective

        if stage == "width":
            self.constraint_decision_stats["width_candidate_infeasible"] += 1
        elif stage == "nnodes":
            self.constraint_decision_stats["nnodes_candidate_infeasible"] += 1
        return 1e9 + fallback_objective

    def _discrete_width_candidates(self, valid_count: int):
        """Discrete width candidates same as add() (max_nnodes·filter by number of valid tokens)."""
        candidate_widths = list(range(10, 151, 10))
        candidate_widths = [w for w in candidate_widths if w <= self.max_nnodes]
        vc = int(valid_count)
        candidate_widths = [w for w in candidate_widths if w >= 1 and w <= vc]
        if not candidate_widths:
            candidate_widths = [min(max(vc, 1), self.max_nnodes)]
        return candidate_widths

    def initialize(self, logits):
        # tree draft
        self.draft_total_time = 0.0
        # tree draft ( draft_time )
        self.expected_draft_total_time = 0.0
        # depth width ( depth width )
        self.prev_final_nnodes = 0  # tree top_index (final_width)
        self.prev_depth = 0  # tree depth
        self.prev_width = 0  # tree width
        # accept length
        self.sum_expected_accepted_length = 1.0
        # accept length
        self.prev_sum_expected_accepted_length = 1.0
        self.prev_target_time = 0.2
        self.prev_per_token_latency = None
        self.prev_per_token_cost = None
        # Depth ( tree )
        self.depth_stats = {}
        self.prev_objective_value = None
        
        # depth width
        # initialize depth per_token_probability_bound
        single_logits = logits[0][-1]  # (vocab_size,)
        
        # 1. per_token_probability_bound 0
        probs = torch.softmax(single_logits, dim=-1)  # (vocab_size,)
        if self.per_token_probability_bound > 0.0:
            probs = torch.where(probs >= self.per_token_probability_bound, probs, torch.zeros_like(probs))
        
        # 2. 0 current_width
        # initialize per_token
        if self.fixed_width is not None:
            # : fixed_width
            self.current_width = min(self.fixed_width, self.max_nnodes)
        else:
            valid_mask = probs > 0
            valid_count = valid_mask.sum().item()
            # add() width
            self.current_width = max(self._discrete_width_candidates(valid_count))
        self.depth_widths.append(self.current_width)
        
        # 3. current_width
        top_logits, ids = torch.topk(probs, k=self.current_width, dim=-1)
        # NOTE: weight_matrix "logits"
        selected_logits = single_logits[ids]
        self.weight_matrix[self.depth, :self.current_width].copy_(selected_logits)
        self.input_ids_matrix[self.depth, :self.current_width].copy_(ids)
        
        rows = torch.arange(self.current_width, device=self.device)
        self.parents_matrix[0, :self.current_width].copy_(rows)
        
        position_id = torch.zeros([self.current_width], dtype=torch.long, device=self.device)
        tri = torch.eye(self.current_width, dtype=torch.int8, device=self.device)
        
        # max_depth 1 initialize
        is_final = (self.depth + 1 >= self.max_depth)
        
        output_dict = {
            "input_ids": ids,   # [DH] logits N (N )
            "position_ids": position_id + 1,   # [DH] depth , initialize 1
            "attention_mask": tri,  # [DH] attention mask is identity matrix
            "parent_last": rows,
            "is_final": is_final
        }
        self.depth += 1
        return output_dict
    
    def add(self, logits):
        self.prev_width = self.current_width  # depth width
        
        # logits (softmax )
        probs = logits[0]  # (self.prev_width, vocab_size)
        
        # width
        # 1. per_token_probability_bound 0
        if self.per_token_probability_bound > 0.0:
            probs = torch.where(probs >= self.per_token_probability_bound, probs, torch.zeros_like(probs))
        
        # 2. , per_path_probability_bound 0
        last_layer_weights = self.weight_matrix[self.depth-1, :self.prev_width].unsqueeze(1)  # (self.prev_width, 1)
        path_probs = probs * last_layer_weights  # (self.prev_width, vocab_size) - = *
        
        # per_path_probability_bound 0
        if self.per_path_probability_bound > 0.0:
            path_probs = torch.where(path_probs >= self.per_path_probability_bound, path_probs, torch.zeros_like(path_probs))
        
        # 3. flat_path_probs 
        flat_path_probs = path_probs.view(-1)  # (self.prev_width * vocab_size)
        
        # 4. width current_width
        if self.fixed_width is not None:
            # : fixed_width
            self.current_width = min(self.fixed_width, self.max_nnodes)
        else:
            valid_mask = flat_path_probs > 0
            valid_count = valid_mask.sum().item()
            candidate_widths = self._discrete_width_candidates(valid_count)

            best_width = candidate_widths[0] if candidate_widths else min(max(valid_count, 1), self.max_nnodes)
            best_objective = float('inf')
            
            # latency cost
            current_latency = self.draft_total_time
            current_cost = self.draft_total_time * self._draft_objective_rate()
            
            # width
            width_algorithm_start_time = time.time()

            # : width torch.topk/ clone
            # (1) topk width 1 prefix sum
            # (2) depth (depth_probs) depth
            prev_depth_probs = []
            for d in range(self.depth):
                w = self.weight_matrix[d, :self.depth_widths[d]]
                prob_sum = w.sum()
                prev_depth_probs.append(float(prob_sum.item()) if isinstance(prob_sum, torch.Tensor) else float(prob_sum))

            max_k = max(candidate_widths) if candidate_widths else 1
            # : max_k flat_path_probs
            max_k = min(int(max_k), int(flat_path_probs.numel()))
            # 0
            if max_k <= 0:
                max_k = 1

            topk_logits_all, topk_idx_all = torch.topk(flat_path_probs, k=max_k, dim=-1)
            topk_prefix_sum = torch.cumsum(topk_logits_all, dim=0)
            
            draft_rate = self._draft_objective_rate()
            target_rate = self._target_objective_rate()
            d2t_time_per_token = float(self.per_token_draft_to_target_transfer_time)
            t2d_time_per_token = float(self.per_token_target_to_draft_transfer_time)

            # width depth
            candidate_width_indices = [max(0, int(w) - 1) for w in candidate_widths]
            candidate_prefix_idx = torch.tensor(
                candidate_width_indices,
                dtype=torch.long,
                device=topk_prefix_sum.device,
            )
            candidate_new_depth_tensor = topk_prefix_sum.index_select(0, candidate_prefix_idx).to(dtype=torch.float32)

            # width expected_accept_length
            expected_sum_tensor = None
            if candidate_widths:
                if prev_depth_probs:
                    prev_depth_tensor = torch.tensor(
                        prev_depth_probs,
                        dtype=torch.float32,
                        device=candidate_new_depth_tensor.device,
                    )
                    prev_depth_matrix = prev_depth_tensor.unsqueeze(0).repeat(len(candidate_widths), 1)
                    all_depth_probs = torch.cat([prev_depth_matrix, candidate_new_depth_tensor.unsqueeze(1)], dim=1)
                else:
                    all_depth_probs = candidate_new_depth_tensor.unsqueeze(1)

                adjusted_depth_probs = torch.zeros_like(all_depth_probs)
                if all_depth_probs.shape[1] > 1:
                    adjusted_depth_probs[:, :-1] = torch.clamp(
                        all_depth_probs[:, :-1] - all_depth_probs[:, 1:],
                        min=0.0,
                    )
                adjusted_depth_probs[:, -1] = torch.clamp(all_depth_probs[:, -1], min=0.0)
                adjusted_depth_probs = torch.nan_to_num(adjusted_depth_probs, nan=0.0, posinf=0.0, neginf=0.0)

                depth_weights = torch.arange(
                    1,
                    adjusted_depth_probs.shape[1] + 1,
                    dtype=torch.float32,
                    device=adjusted_depth_probs.device,
                ).unsqueeze(0)
                expected_sum_tensor = torch.sum(adjusted_depth_probs * depth_weights, dim=1)
                expected_sum_tensor = torch.nan_to_num(expected_sum_tensor, nan=0.0, posinf=0.0, neginf=0.0)
                expected_sum_tensor = torch.where(
                    expected_sum_tensor > 0.0,
                    expected_sum_tensor,
                    torch.ones_like(expected_sum_tensor),
                )

            # target / (profile , nnodes )
            # width / est_nnodes .
            base_nnodes = self.prev_final_nnodes if self.prev_final_nnodes > 0 else self.max_nnodes
            est_nnodes = min(self.max_nnodes, base_nnodes)
            est_target_time = self._lookup_target_time_cached(
                nnodes=int(est_nnodes),
                default_time=0.2 * float(self.prev_final_nnodes) / float(self.max_nnodes),
            )

            if candidate_widths and expected_sum_tensor is not None:
                next_predicted_time_tensor = self._get_width_predicted_times_tensor(
                    candidate_widths,
                    expected_sum_tensor.device,
                )
                denom_expected_tensor = (
                    expected_sum_tensor * float(self.accept_length_scale)
                    * max(0.0, 1.0 - float(self.accept_length_margin))
                )
                denom_sum_tensor = denom_expected_tensor + 1.0
                transfer_time_tensor = (
                    d2t_time_per_token * float(est_nnodes)
                    + t2d_time_per_token * denom_expected_tensor
                )
                total_latency_tensor = (
                    float(current_latency)
                    + next_predicted_time_tensor
                    + transfer_time_tensor
                    + float(est_target_time)
                )

                bytes_per_gb = float(1024 ** 3)
                if self._uses_any_cost_objective() and bytes_per_gb > 0:
                    d2t_bytes = float(est_nnodes) * float(self.per_token_draft_to_target_bytes)
                    t2d_bytes_tensor = denom_expected_tensor * float(self.per_token_target_to_draft_bytes)
                    inbound_cost = (d2t_bytes / bytes_per_gb) * float(self.user_communication_cost_per_gb)
                    user_outbound_cost_tensor = (
                        t2d_bytes_tensor / bytes_per_gb
                    ) * float(self.user_communication_cost_per_gb)
                    cloud_outbound_cost_tensor = (
                        t2d_bytes_tensor / bytes_per_gb
                    ) * float(self.cloud_outbound_cost_per_gb)
                    if self._uses_total_cost_objective():
                        transfer_cost_tensor = torch.clamp(
                            inbound_cost + user_outbound_cost_tensor + cloud_outbound_cost_tensor,
                            min=0.0,
                        )
                    elif self._uses_api_cost_objective():
                        transfer_cost_tensor = torch.clamp(
                            user_outbound_cost_tensor + cloud_outbound_cost_tensor,
                            min=0.0,
                        )
                    else:
                        transfer_cost_tensor = torch.zeros_like(denom_expected_tensor)
                else:
                    transfer_cost_tensor = torch.zeros_like(denom_expected_tensor)

                total_cost_tensor = (
                    float(current_cost)
                    + (next_predicted_time_tensor * float(draft_rate))
                    + transfer_cost_tensor
                    + (float(est_target_time) * float(target_rate))
                )
                valid_mask = (denom_sum_tensor > 0) & (total_latency_tensor > 0)

                if self.objective_selection_mode == "constraint":
                    metric_per_token_tensor = torch.where(
                        valid_mask,
                        total_cost_tensor / denom_sum_tensor,
                        torch.full_like(denom_sum_tensor, float("inf")),
                    )
                    per_token_latency_tensor = torch.where(
                        valid_mask,
                        total_latency_tensor / denom_sum_tensor,
                        torch.full_like(denom_sum_tensor, float("inf")),
                    )
                    current_tps_tensor = torch.where(
                        valid_mask,
                        1.0 / torch.clamp(per_token_latency_tensor, min=1e-9),
                        torch.zeros_like(per_token_latency_tensor),
                    )

                    if self.constraint_target == "tps":
                        if self.min_tps_constraint is not None and self.min_tps_constraint > 0:
                            feasible_mask = valid_mask & (
                                current_tps_tensor >= float(self.min_tps_constraint)
                            )
                        else:
                            feasible_mask = valid_mask
                        objective_tensor = torch.where(
                            feasible_mask,
                            metric_per_token_tensor,
                            torch.where(
                                valid_mask,
                                torch.full_like(per_token_latency_tensor, 1e9) + per_token_latency_tensor,
                                torch.full_like(per_token_latency_tensor, float("inf")),
                            ),
                        )
                    else:
                        if self.metric_constraint_per_token is not None and self.metric_constraint_per_token > 0:
                            feasible_mask = valid_mask & (
                                metric_per_token_tensor <= float(self.metric_constraint_per_token)
                            )
                        else:
                            feasible_mask = valid_mask
                        objective_tensor = torch.where(
                            feasible_mask,
                            per_token_latency_tensor,
                            torch.where(
                                valid_mask,
                                torch.full_like(metric_per_token_tensor, 1e9) + metric_per_token_tensor,
                                torch.full_like(metric_per_token_tensor, float("inf")),
                            ),
                        )

                    feasible_count = int(torch.sum(feasible_mask.to(torch.int32)).item())
                    valid_count = int(torch.sum(valid_mask.to(torch.int32)).item())
                    infeasible_count = max(0, valid_count - feasible_count)
                    self.constraint_decision_stats["width_candidate_feasible"] += feasible_count
                    self.constraint_decision_stats["width_candidate_infeasible"] += infeasible_count

                    best_idx = int(torch.argmin(objective_tensor).item())
                    best_objective = float(objective_tensor[best_idx].item())
                    best_width = int(candidate_widths[best_idx])
                else:
                    current_tps_tensor = torch.where(
                        valid_mask,
                        denom_sum_tensor / total_latency_tensor,
                        torch.zeros_like(denom_sum_tensor),
                    )
                    current_obj_per_token_tensor = torch.where(
                        valid_mask,
                        total_cost_tensor / denom_sum_tensor,
                        torch.full_like(denom_sum_tensor, float("inf")),
                    )
                    ref_tps = max(1e-9, float(self.reference_tps))
                    ref_obj = max(1e-12, float(self.reference_objective_per_token))
                    normalized_tps_tensor = torch.where(
                        current_tps_tensor > 0,
                        torch.full_like(current_tps_tensor, float(ref_tps)) / current_tps_tensor,
                        torch.full_like(current_tps_tensor, float("inf")),
                    )
                    normalized_cost_tensor = current_obj_per_token_tensor / float(ref_obj)
                    alpha = float(self._sensitivity_alpha())
                    objective_tensor = (alpha * normalized_cost_tensor) + ((1.0 - alpha) * normalized_tps_tensor)
                    objective_tensor = torch.where(
                        valid_mask,
                        objective_tensor,
                        torch.full_like(objective_tensor, float("inf")),
                    )
                    best_idx = int(torch.argmin(objective_tensor).item())
                    best_objective = float(objective_tensor[best_idx].item())
                    best_width = int(candidate_widths[best_idx])
            
            # width
            width_algorithm_time = time.time() - width_algorithm_start_time
            self.width_algorithm_times.append(width_algorithm_time)
            if self.objective_selection_mode == "constraint" and math.isfinite(best_objective):
                if best_objective >= 1e9:
                    self.constraint_decision_stats["width_selected_fallback"] += 1
                else:
                    self.constraint_decision_stats["width_selected_feasible"] += 1
            
            self.current_width = best_width
        
        self.depth_widths.append(self.current_width)
        
        # 5. current_width
        # : topk(max_k) prefix slice
        if self.fixed_width is None and 'topk_logits_all' in locals() and topk_logits_all is not None and self.current_width <= int(topk_logits_all.numel()):
            global_top_logits = topk_logits_all[: self.current_width]
            global_top_idx = topk_idx_all[: self.current_width]
        else:
            global_top_logits, global_top_idx = torch.topk(flat_path_probs, k=self.current_width, dim=-1)
        
        # global_top_idx (self.prev_width, vocab_size)
        vocab_size = probs.size(1)
        parents = global_top_idx // vocab_size
        input_ids = global_top_idx % vocab_size  # ID
        
        # parents self.prev_width
        parents = torch.clamp(parents, 0, self.prev_width - 1)
        
        self.parents_matrix[self.depth, :self.current_width].copy_(parents)
        self.weight_matrix[self.depth, :self.current_width].copy_(global_top_logits)
        self.input_ids_matrix[self.depth, :self.current_width].copy_(input_ids)
        
        rows = torch.arange(self.current_width, device=self.device)
        kv_cache_mask = torch.zeros([self.current_width, self.prev_width], dtype=torch.int8, device=self.device) 
        # [DH] (current_width, self.prev_width) matrix. depth parent .
        # ( index 3 (0, 3) 1 )
        kv_cache_mask[rows, parents] = 1
        
        # kv_mask
        if self.depth == 1:
            kv_mask = kv_cache_mask
        else:
            # kv_mask parent
            prev_kv_mask = getattr(self, '_prev_kv_mask', torch.zeros([self.prev_width, 0], dtype=torch.int8, device=self.device))
            kv_mask = torch.cat([prev_kv_mask[parents], kv_cache_mask], dim=1)
        
        self._prev_kv_mask = kv_mask
        
        tri = torch.eye(self.current_width, dtype=torch.int8, device=self.device)
        attention_mask = torch.cat([kv_mask, tri], dim=1)    # [DH] child attention mask
        
        position_id = torch.zeros([self.current_width], dtype=torch.long, device=self.device)
        output_dict = {
            "input_ids": input_ids,  # [DH] self.prev_width current_width
            "position_ids": position_id + (self.depth + 1),  # [DH] depth
            "attention_mask": attention_mask,
            "parent_last": parents,
            "is_final": False
        }
        self.depth += 1
        return output_dict
    
    def generate_attention_mask(self, parents, final_width):
        attention_mask = torch.eye(final_width, dtype=torch.int8, device=self.device)
        grandp = parents.clone()
        for _ in range(self.depth - 2):
            rows = torch.arange(final_width, device=self.device)
            attention_mask[rows, grandp] = 1
            grandp = grandp[parents]
        return attention_mask

    def get_objective_value(self, valid_weights):
        """
        Function that finds final_nnodes minimizing objective_value
        
        Args:
            valid_weights: weight tensor list for each depth
        
        Returns:
            tuple: (objective_value, top_index, weight, per_token_latency, per_token_cost, 
                    current_target_time, current_per_token_latency, current_per_token_cost)
        """
        flat_weights = torch.cat(valid_weights)
        max_available_nodes = len(flat_weights)
        
        # fixed_nnodes True max_nnodes 
        if self.fixed_nnodes:
            # max_nnodes
            fixed_nnodes_val = min(self.max_nnodes, max_available_nodes) if max_available_nodes > 0 else 1
            candidate_nnodes = [fixed_nnodes_val]
        else:
            # final_nnodes
            candidate_nnodes = list(range(10, 151, 10))
            candidate_nnodes = [n for n in candidate_nnodes if n <= max_available_nodes]
            if not candidate_nnodes:
                candidate_nnodes = [max_available_nodes] if max_available_nodes > 0 else [1]
        
        # target
        prev_target_time = self.prev_target_time
        
        # target 
        if self.prev_final_nnodes > 0 and self.target_profile_data is not None and prev_target_time == 0.2:
            prev_target_time = self._lookup_target_time_cached(
                nnodes=int(self.prev_final_nnodes),
                default_time=0.2,
            )
            self.prev_target_time = prev_target_time
        
        # per_token_latency per_token_cost 
        # accept_length_scale expected_accept_length .
        prev_scaled_sum = self._apply_accept_conservative_margin(
            float(self.prev_sum_expected_accepted_length) * float(self.accept_length_scale)
        )
        safe_prev_sum = max(1.0, prev_scaled_sum) if prev_scaled_sum > 0 else 1.0
        denom_prev_sum = safe_prev_sum + 1.0
        if self.prev_per_token_latency is not None:
            prev_per_token_latency = self.prev_per_token_latency
        else:
            prev_transfer_time = (
                self.per_token_draft_to_target_transfer_time * self.prev_final_nnodes
                + self.per_token_target_to_draft_transfer_time * denom_prev_sum
            )
            prev_per_token_latency = (self.prev_draft_total_time + prev_transfer_time + prev_target_time) / denom_prev_sum
        
        if self.prev_per_token_cost is not None:
            prev_per_token_cost = self.prev_per_token_cost
        else:
            prev_draft_cost = self.prev_draft_total_time * self._draft_objective_rate()
            prev_per_token_cost = (
                prev_draft_cost
                + self._transfer_objective_cost_from_tokens(
                    d2t_tokens=float(self.prev_final_nnodes),
                    t2d_tokens=float(denom_prev_sum),
                )
                + prev_target_time * self._target_objective_rate()
            ) / denom_prev_sum
        
        # final_nnodes
        nnodes_algorithm_start_time = time.time()
        
        # objective_value
        best_objective_value = float('inf')
        best_final_nnodes = candidate_nnodes[0]
        best_top_index = None
        best_weight = 0.0
        best_per_token_latency = 0.0
        best_per_token_cost = 0.0
        best_current_target_time = 0.2
        best_sum_expected_accepted_length = 1.0
        
        # : nnodes torch.topk/ / ,
        # (1) k topk 1
        # (2) topk (rank 0..k-1) " depth " 1
        # (3) depth prefix sum k O(depth) sum_expected_accepted_length
        max_k = max(candidate_nnodes) if candidate_nnodes else 1
        max_k = min(int(max_k), int(flat_weights.numel()))
        if max_k <= 0:
            max_k = 1
        top_weights_all, top_index_all = torch.topk(flat_weights, k=max_k, dim=-1)
        top_weight_prefix = torch.cumsum(top_weights_all, dim=0)

        # rank depth id (cumsum widths)
        # boundaries: [w0, w0+w1, ...] ( depth )
        if self.depth_widths and self.depth > 1:
            cum = 0
            boundaries_list = []
            for d in range(self.depth - 1):
                cum += int(self.depth_widths[d])
                boundaries_list.append(cum)
            boundaries = torch.tensor(boundaries_list, device=top_index_all.device, dtype=top_index_all.dtype)
            depth_ids = torch.bucketize(top_index_all, boundaries, right=False)  # 0..depth-1
        else:
            depth_ids = torch.zeros_like(top_index_all)

        # depth prefix sum (rank ) : prefix_by_depth[d][r] = r+1 depth=d
        # depth (max_depth <= 10), max_k (depth * max_k)
        top_weights_all_f = top_weights_all.to(dtype=torch.float32)
        prefix_by_depth = []
        for d in range(self.depth):
            mask = (depth_ids == d).to(dtype=torch.float32)
            prefix_by_depth.append(torch.cumsum(top_weights_all_f * mask, dim=0))
        # .item() GPU
        # prefix CPU .
        prefix_by_depth_tensor = None
        if prefix_by_depth:
            prefix_by_depth_tensor = torch.stack(prefix_by_depth, dim=0)

        candidate_nnodes_int = [int(n) for n in candidate_nnodes]
        candidate_expected_sum_by_nnodes = {}
        if candidate_nnodes_int:
            candidate_rank_indices = torch.tensor(
                [max(0, k - 1) for k in candidate_nnodes_int],
                dtype=torch.long,
                device=top_index_all.device,
            )
            if prefix_by_depth_tensor is not None:
                # [candidate, depth] depth
                candidate_depth_probs = prefix_by_depth_tensor.index_select(1, candidate_rank_indices).transpose(0, 1)
            else:
                candidate_depth_probs = torch.zeros(
                    (len(candidate_nnodes_int), self.depth),
                    dtype=torch.float32,
                    device=top_index_all.device,
                )

            eps = 1e-5
            candidate_depth_probs = torch.nan_to_num(candidate_depth_probs, nan=0.0, posinf=0.0, neginf=0.0)
            candidate_depth_probs = torch.where(
                (candidate_depth_probs >= -eps) & (candidate_depth_probs < 0.0),
                torch.zeros_like(candidate_depth_probs),
                candidate_depth_probs,
            )
            candidate_depth_probs = torch.where(
                (candidate_depth_probs > 1.0) & (candidate_depth_probs <= 1.0 + eps),
                torch.ones_like(candidate_depth_probs),
                candidate_depth_probs,
            )
            candidate_depth_probs = torch.where(
                (candidate_depth_probs >= 0.0) & (candidate_depth_probs <= 1.0),
                candidate_depth_probs,
                torch.zeros_like(candidate_depth_probs),
            )

            adjusted_depth_probs = torch.zeros_like(candidate_depth_probs)
            if self.depth > 1:
                adjusted_depth_probs[:, :-1] = torch.clamp(
                    candidate_depth_probs[:, :-1] - candidate_depth_probs[:, 1:],
                    min=0.0,
                )
            adjusted_depth_probs[:, -1] = torch.clamp(candidate_depth_probs[:, -1], min=0.0)

            depth_weights = torch.arange(
                1,
                self.depth + 1,
                dtype=torch.float32,
                device=candidate_depth_probs.device,
            ).unsqueeze(0)
            candidate_expected_sum = torch.sum(adjusted_depth_probs * depth_weights, dim=1)
            candidate_expected_sum = torch.nan_to_num(candidate_expected_sum, nan=0.0, posinf=0.0, neginf=0.0)
            candidate_expected_sum = torch.where(
                candidate_expected_sum > 0.0,
                candidate_expected_sum,
                torch.ones_like(candidate_expected_sum),
            )
            if self.objective_selection_mode in {"constraint", "blend"}:
                d2t_tokens_tensor = torch.tensor(
                    candidate_nnodes_int,
                    dtype=torch.float32,
                    device=candidate_depth_probs.device,
                )
                denom_expected_tensor = (
                    candidate_expected_sum
                    * float(self.accept_length_scale)
                    * max(0.0, 1.0 - float(self.accept_length_margin))
                )
                denom_sum_tensor = denom_expected_tensor + 1.0
                target_time_list = [
                    float(self._lookup_target_time_cached(nnodes=int(n), default_time=0.2))
                    for n in candidate_nnodes_int
                ]
                target_time_tensor = torch.tensor(
                    target_time_list,
                    dtype=torch.float32,
                    device=candidate_depth_probs.device,
                )
                transfer_time_tensor = (
                    float(self.per_token_draft_to_target_transfer_time) * d2t_tokens_tensor
                    + float(self.per_token_target_to_draft_transfer_time) * denom_sum_tensor
                )
                total_latency_tensor = (
                    float(self.draft_total_time)
                    + transfer_time_tensor
                    + target_time_tensor
                )
                bytes_per_gb = float(1024 ** 3)
                if self._uses_any_cost_objective() and bytes_per_gb > 0:
                    d2t_bytes_tensor = d2t_tokens_tensor * float(self.per_token_draft_to_target_bytes)
                    t2d_bytes_tensor = denom_sum_tensor * float(self.per_token_target_to_draft_bytes)
                    inbound_cost_tensor = (
                        d2t_bytes_tensor / bytes_per_gb
                    ) * float(self.user_communication_cost_per_gb)
                    user_outbound_cost_tensor = (
                        t2d_bytes_tensor / bytes_per_gb
                    ) * float(self.user_communication_cost_per_gb)
                    cloud_outbound_cost_tensor = (
                        t2d_bytes_tensor / bytes_per_gb
                    ) * float(self.cloud_outbound_cost_per_gb)
                    if self._uses_total_cost_objective():
                        transfer_cost_tensor = torch.clamp(
                            inbound_cost_tensor + user_outbound_cost_tensor + cloud_outbound_cost_tensor,
                            min=0.0,
                        )
                    elif self._uses_api_cost_objective():
                        transfer_cost_tensor = torch.clamp(
                            user_outbound_cost_tensor + cloud_outbound_cost_tensor,
                            min=0.0,
                        )
                    else:
                        transfer_cost_tensor = torch.zeros_like(denom_sum_tensor)
                else:
                    transfer_cost_tensor = torch.zeros_like(denom_sum_tensor)

                total_cost_tensor = (
                    float(self.draft_total_time) * float(self._draft_objective_rate())
                    + transfer_cost_tensor
                    + target_time_tensor * float(self._target_objective_rate())
                )
                valid_mask = (denom_sum_tensor > 0) & (total_latency_tensor > 0)

                if self.objective_selection_mode == "constraint":
                    metric_per_token_tensor = torch.where(
                        valid_mask,
                        total_cost_tensor / denom_sum_tensor,
                        torch.full_like(denom_sum_tensor, float("inf")),
                    )
                    per_token_latency_tensor = torch.where(
                        valid_mask,
                        total_latency_tensor / denom_sum_tensor,
                        torch.full_like(denom_sum_tensor, float("inf")),
                    )
                    current_tps_tensor = torch.where(
                        valid_mask,
                        1.0 / torch.clamp(per_token_latency_tensor, min=1e-9),
                        torch.zeros_like(per_token_latency_tensor),
                    )

                    if self.constraint_target == "tps":
                        if self.min_tps_constraint is not None and self.min_tps_constraint > 0:
                            feasible_mask = valid_mask & (
                                current_tps_tensor >= float(self.min_tps_constraint)
                            )
                        else:
                            feasible_mask = valid_mask
                        objective_tensor = torch.where(
                            feasible_mask,
                            metric_per_token_tensor,
                            torch.where(
                                valid_mask,
                                torch.full_like(per_token_latency_tensor, 1e9) + per_token_latency_tensor,
                                torch.full_like(per_token_latency_tensor, float("inf")),
                            ),
                        )
                    else:
                        if self.metric_constraint_per_token is not None and self.metric_constraint_per_token > 0:
                            feasible_mask = valid_mask & (
                                metric_per_token_tensor <= float(self.metric_constraint_per_token)
                            )
                        else:
                            feasible_mask = valid_mask
                        objective_tensor = torch.where(
                            feasible_mask,
                            per_token_latency_tensor,
                            torch.where(
                                valid_mask,
                                torch.full_like(metric_per_token_tensor, 1e9) + metric_per_token_tensor,
                                torch.full_like(metric_per_token_tensor, float("inf")),
                            ),
                        )

                    feasible_count = int(torch.sum(feasible_mask.to(torch.int32)).item())
                    valid_count = int(torch.sum(valid_mask.to(torch.int32)).item())
                    infeasible_count = max(0, valid_count - feasible_count)
                    self.constraint_decision_stats["nnodes_candidate_feasible"] += feasible_count
                    self.constraint_decision_stats["nnodes_candidate_infeasible"] += infeasible_count
                else:
                    current_tps_tensor = torch.where(
                        valid_mask,
                        denom_sum_tensor / total_latency_tensor,
                        torch.zeros_like(denom_sum_tensor),
                    )
                    current_obj_per_token_tensor = torch.where(
                        valid_mask,
                        total_cost_tensor / denom_sum_tensor,
                        torch.full_like(denom_sum_tensor, float("inf")),
                    )
                    ref_tps = max(1e-9, float(self.reference_tps))
                    ref_obj = max(1e-12, float(self.reference_objective_per_token))
                    normalized_tps_tensor = torch.where(
                        current_tps_tensor > 0,
                        torch.full_like(current_tps_tensor, float(ref_tps)) / current_tps_tensor,
                        torch.full_like(current_tps_tensor, float("inf")),
                    )
                    normalized_cost_tensor = current_obj_per_token_tensor / float(ref_obj)
                    alpha = float(self._sensitivity_alpha())
                    objective_tensor = (alpha * normalized_cost_tensor) + ((1.0 - alpha) * normalized_tps_tensor)
                    objective_tensor = torch.where(
                        valid_mask,
                        objective_tensor,
                        torch.full_like(objective_tensor, float("inf")),
                    )

                best_idx = int(torch.argmin(objective_tensor).item())
                best_final_nnodes = int(candidate_nnodes_int[best_idx])
                best_top_index = top_index_all[:best_final_nnodes]
                best_weight = top_weight_prefix[best_final_nnodes - 1]
                best_objective_value = float(objective_tensor[best_idx].item())
                best_sum_expected_accepted_length = float(candidate_expected_sum[best_idx].item())
                best_current_target_time = float(target_time_tensor[best_idx].item())
                best_denom_sum = float(denom_sum_tensor[best_idx].item())
                best_total_latency = float(total_latency_tensor[best_idx].item())
                best_total_cost = float(total_cost_tensor[best_idx].item())
                best_per_token_latency = best_total_latency / max(1e-9, best_denom_sum)
                best_per_token_cost = best_total_cost / max(1e-9, best_denom_sum)
                candidate_expected_sum_by_nnodes = {}
            else:
                for idx, n in enumerate(candidate_nnodes_int):
                    candidate_expected_sum_by_nnodes[n] = float(candidate_expected_sum[idx].item())
        
        # final_nnodes
        nnodes_algorithm_time = time.time() - nnodes_algorithm_start_time
        self.nnodes_algorithm_times.append(nnodes_algorithm_time)
        if self.objective_selection_mode == "constraint" and math.isfinite(best_objective_value):
            if best_objective_value >= 1e9:
                self.constraint_decision_stats["nnodes_selected_fallback"] += 1
            else:
                self.constraint_decision_stats["nnodes_selected_feasible"] += 1
        
        # best_top_index None 
        if best_top_index is None:
            # , topk
            if len(candidate_nnodes) > 0 and max_available_nodes > 0:
                final_nnodes_val = min(candidate_nnodes[0], max_available_nodes)
                # topk
                if 'top_index_all' in locals() and top_index_all is not None and final_nnodes_val <= int(top_index_all.numel()):
                    best_top_index = top_index_all[:final_nnodes_val]
                else:
                    _, best_top_index = torch.topk(flat_weights, k=final_nnodes_val, dim=-1)
                best_final_nnodes = final_nnodes_val
                best_weight = flat_weights[best_top_index].sum()
            else:
                # flat_weights
                best_top_index = torch.tensor([], dtype=torch.long, device=flat_weights.device)
                best_final_nnodes = 1
                best_weight = 0.0
        
        self.final_nnodes = best_final_nnodes
        self.sum_expected_accepted_length = best_sum_expected_accepted_length
        
        # per_token_latency per_token_cost 
        # NOTE: prev_per_token_latency/prev_per_token_cost,
        # accept_length_scale diff .
        scaled_best_sum = self._apply_accept_conservative_margin(
            float(best_sum_expected_accepted_length) * float(self.accept_length_scale)
        )
        safe_sum = max(1.0, scaled_best_sum) if scaled_best_sum > 0 else 1.0
        denom_best_sum = safe_sum + 1.0
        best_transfer_time = (
            self.per_token_draft_to_target_transfer_time * self.final_nnodes
            + self.per_token_target_to_draft_transfer_time * denom_best_sum
        )
        current_per_token_latency = (self.draft_total_time + best_transfer_time + best_current_target_time) / denom_best_sum
        current_draft_cost = self.draft_total_time * self._draft_objective_rate()
        current_per_token_cost = (
            current_draft_cost
            + self._transfer_objective_cost_from_tokens(
                d2t_tokens=float(self.final_nnodes),
                t2d_tokens=float(denom_best_sum),
            )
            + best_current_target_time * self._target_objective_rate()
        ) / denom_best_sum
        
        return best_objective_value, best_top_index, best_weight, best_per_token_latency, best_per_token_cost, best_current_target_time, current_per_token_latency, current_per_token_cost

    def update(self, logits, print_tree=False, tokenizer=None, draft_time=None, draft_time_scale: float = 1.0):
        self.last_finalize_time_sec = 0.0
        if self.depth == 0:
            result = self.initialize(logits)
            if print_tree:
                self.print_tree_structure_hierarchical(tokenizer)
            return result
        
        # add() GPU ( CUDA )
        if self._should_sync_timing(logits):
            torch.cuda.synchronize()
        
        # add()
        start_time = time.time()
        outputs = self.add(logits)
        
        # GPU : synchronize() add() CUDA
        if self._should_sync_timing(logits):
            torch.cuda.synchronize()
        
        # synchronize() (add() CUDA )
        elapsed_time = time.time() - start_time
        
        # tree draft
        self.prev_draft_total_time = self.draft_total_time
        observed_draft_time = 0.0
        predicted_prev_call_time = float(self._predict_next_time_for_width(self.prev_width))
        # draft_time None or elapsed_time is None 
        if draft_time is not None:
            self.draft_total_time += draft_time
            observed_draft_time = float(draft_time)
        else:
            # elapsed_time
            self.draft_total_time += elapsed_time
            observed_draft_time = float(elapsed_time)
        self.last_observed_draft_time_sec = max(0.0, observed_draft_time)
        self._record_proactive_prediction_error_ratio(
            width=int(max(1, self.prev_width)),
            predicted_sec=predicted_prev_call_time,
            observed_sec=observed_draft_time,
        )
        
        # width
        # - "draft_time" eval_autodraft_draft.py draft_model.model() forward pass,
        # update logits (= depth width).
        # - add() update self.prev_width = self.current_width,
        # draft_time self.prev_width .
        current_width = self.current_width
        
        # tree draft
        # width model_call_avg_time_ms (draft_time )
        if self.profile_data is not None:
            width_str = str(self.prev_width)
            if width_str in self.profile_data:
                avg_time_ms = self.profile_data[width_str].get("model_call_avg_time_ms", 0.0)
                # Update the draft-time scale from draft_time / predicted_time_ms.
                scale = float(draft_time_scale) if draft_time_scale is not None else 1.0
                self.expected_draft_total_time += (avg_time_ms * scale) / 1000.0
            else:
                # width draft_time
                if draft_time is not None:
                    self.expected_draft_total_time += draft_time
                else:
                    # elapsed_time
                    self.expected_draft_total_time += elapsed_time
        else:
            # draft_time
            if draft_time is not None:
                self.expected_draft_total_time += draft_time
            else:
                # elapsed_time
                self.expected_draft_total_time += elapsed_time
        # width_times add() draft_model.model()
        # update draft_time None .
        if draft_time is not None:
            width_key = int(self.prev_width)
            if width_key not in self.width_times:
                self.width_times[width_key] = []
            self.width_times[width_key].append(float(draft_time))
            self._width_time_running_sum[width_key] = (
                float(self._width_time_running_sum.get(width_key, 0.0)) + float(draft_time)
            )
            self._width_time_running_count[width_key] = (
                int(self._width_time_running_count.get(width_key, 0)) + 1
            )
            self._invalidate_width_prediction_cache()
        # width
        self.last_observed_width = int(max(1, self.prev_width))
        
        if print_tree:
            self.print_tree_structure_hierarchical(tokenizer)
        
        # breakpoint()
        
        # depth width ,
        total_nodes = sum(self.depth_widths[:self.depth-1])
          
        # weight_matrix
        # add() self.depth += 1 , depth self.depth - 1
        # range(self.depth) depth
        valid_weights = []
        for d in range(self.depth):
            valid_weights.append(self.weight_matrix[d, :self.depth_widths[d]])
        
        self.prev_sum_expected_accepted_length = self.sum_expected_accepted_length
        
        # objective_value
        objective_value, top_index, weight, per_token_latency, per_token_cost, current_target_time, current_per_token_latency, current_per_token_cost = self.get_objective_value(valid_weights)
        
        # fixed-nnodes , tree_temp.py sum_expected_accepted_length
        flat_weights = torch.cat(valid_weights)
        if self.fixed_nnodes and self.final_nnodes >= len(flat_weights):
            # tree_temp.py :
            depth_probs = [w.sum() for w in valid_weights]
            adjusted_depth_probs = []
            for i in range(len(depth_probs)):
                if i == len(depth_probs) - 1:
                    adjusted_depth_probs.append(depth_probs[i])
                else:
                    adjusted_depth_probs.append(max(0.0, depth_probs[i] - depth_probs[i + 1]))
            # (depth+1)
            if adjusted_depth_probs:
                expected_sum = sum(adj_prob * float(d + 1) for d, adj_prob in enumerate(adjusted_depth_probs))
                if math.isnan(expected_sum) or expected_sum <= 0:
                    print(f"[WARNING] expected_sum is invalid (NaN or <= 0): {expected_sum}, setting self.sum_expected_accepted_length to 1.0 (depth={self.depth}, final_nnodes={self.final_nnodes})")
                    self.sum_expected_accepted_length = 1.0
                else:
                    self.sum_expected_accepted_length = float(expected_sum)
            else:
                print(f"[WARNING] adjusted_depth_probs is empty, setting self.sum_expected_accepted_length to 1.0 (depth={self.depth}, final_nnodes={self.final_nnodes})")
                self.sum_expected_accepted_length = 1.0
        # print(f"per token latency: {per_token_latency}, per token cost: {per_token_cost}, objective value: {objective_value}")
        
        self.prev_target_time = current_target_time
        self.prev_per_token_latency = current_per_token_latency
        self.prev_per_token_cost = current_per_token_cost
        
        # Depth ( depth )
        current_depth = self.depth
        if current_depth not in self.depth_stats:
            self.depth_stats[current_depth] = {
                "prev_sum_expected_accepted_length": [],
                "sum_expected_accepted_length": [],
                "per_token_latency": [],
                "per_token_cost": [],
                "objective_value": []
            }
        
        self.depth_stats[current_depth]["prev_sum_expected_accepted_length"].append(float(self.prev_sum_expected_accepted_length))
        self.depth_stats[current_depth]["sum_expected_accepted_length"].append(float(self.sum_expected_accepted_length))
        self.depth_stats[current_depth]["per_token_latency"].append(float(per_token_latency))
        self.depth_stats[current_depth]["per_token_cost"].append(float(per_token_cost))
        self.depth_stats[current_depth]["objective_value"].append(float(objective_value))

        # AutoDraft: depth
        force_stop = self.stop_flag is not None and self.stop_flag.is_set()
        prev_objective_value = self.prev_objective_value if self.prev_objective_value is not None else objective_value
        proactive_policy_active = (
            self.stop_flag is not None
            and not force_stop
            and not self.proactive_disable_budget
            and (
                self.proactive_continue_event is None
                or not self.proactive_continue_event.is_set()
            )
        )
        if proactive_policy_active:
            # proactive : reply / depth .
            # reply continue_event set draft tree .
            budget_sec = self.proactive_time_budget_sec
            if budget_sec is not None:
                predicted_next_time_sec = self._predict_next_time_for_width_conservative(self.current_width)
                predicted_elapsed = (
                    self.expected_draft_total_time
                    if self.expected_draft_total_time > 0
                    else self.draft_total_time
                )
                expand_decision, expected_gain_sec, expected_loss_sec = self._evaluate_proactive_expand_value(
                    predicted_elapsed_sec=predicted_elapsed,
                    predicted_next_sec=predicted_next_time_sec,
                    budget_sec=budget_sec,
                )
                self._record_proactive_expand_decision(
                    expand_decision,
                    self.depth,
                    expected_gain_sec,
                    expected_loss_sec,
                )
                # " " " " .
                if (
                    expand_decision == "pause"
                    or predicted_elapsed + predicted_next_time_sec >= budget_sec
                ):
                    wait_start = time.time()
                    while (
                        not self.stop_flag.is_set()
                        and (
                            self.proactive_continue_event is None
                            or not self.proactive_continue_event.is_set()
                        )
                    ):
                        # cancel(use=false) continue(use=true)
                        self.stop_flag.wait(0.002)
                    self.proactive_budget_wait_sec += max(0.0, float(time.time() - wait_start))
                    force_stop = self.stop_flag.is_set()
            else:
                # fallback
                scaled_sum = self._apply_accept_conservative_margin(
                    float(self.sum_expected_accepted_length) * float(self.accept_length_scale)
                )
                safe_sum = max(1.0, scaled_sum) if scaled_sum > 0 else 1.0
                denom_sum = safe_sum + 1.0
                target_budget = float(current_target_time) + (self.per_token_target_to_draft_transfer_time * denom_sum)
                predicted_next_time_sec = self._predict_next_time_for_width(self.current_width)
                predicted_elapsed = self.expected_draft_total_time if self.expected_draft_total_time > 0 else self.draft_total_time
                if predicted_elapsed + predicted_next_time_sec > target_budget:
                    wait_time = max(0.0, target_budget - predicted_elapsed)
                    self.stop_flag.wait(wait_time)
                    force_stop = self.stop_flag.is_set()
        can_expand = (not force_stop) and (self.depth < self.max_depth)
        if self.opt_tree and weight > self.weight and can_expand:
            self.weight = weight
        elif can_expand and self.depth == 2:  # depth=2
            self.weight = weight
        elif can_expand and self.fixed_depth:  # fixed_depth=True max_depth
            self.weight = weight
        elif can_expand and (not self.fixed_depth) and (prev_objective_value - objective_value) > 0:  # fixed_depth=False objective
            self.weight = weight
        else:   # [DH] depth .
            finalize_start_time = time.time()
            # breakpoint()
            # top_index
            total_nodes = sum(self.depth_widths[:self.depth])
            if force_stop:
                # proactive
                top_index = torch.tensor([0], dtype=torch.long, device=self.device)
                self.final_nnodes = 1
                self.sum_expected_accepted_length = 1.0
            if len(top_index) > 0:
                max_valid_index = total_nodes - 1
                # top_index
                top_index = torch.clamp(top_index, 0, max_valid_index)
            
            # top_index depth column
            rows = torch.zeros_like(top_index)
            cols = top_index.clone()
            cumsum = 0
            for d in range(self.depth):
                if d < len(self.depth_widths):
                    mask = (top_index >= cumsum) & (top_index < cumsum + self.depth_widths[d])
                    rows[mask] = d  # [DH] depth
                    cols[mask] = top_index[mask] - cumsum  # [DH] depth column
                    cumsum += self.depth_widths[d]
                else:
                    break
            
            # rows max_depth
            rows = torch.clamp(rows, 0, min(self.max_depth - 1, len(self.depth_widths) - 1))
            # cols depth width 
            max_cols = torch.tensor([self.depth_widths[d] if d < len(self.depth_widths) else self.max_nnodes 
                                     for d in range(self.max_depth)], device=self.device)
            # row max_col
            row_max_cols = max_cols[rows]
            # row_max_cols 0 max -1 1
            row_max_cols = torch.clamp(row_max_cols, min=1)
            max_per_row = row_max_cols - 1
            # torch.clamp (Tensor, Number, Tensor) Tensor- clamp
            cols = torch.maximum(cols, torch.zeros_like(cols))
            cols = torch.minimum(cols, max_per_row)
            # max_nnodes
            cols = torch.minimum(cols, torch.full_like(cols, self.max_nnodes - 1))
            
            # breakpoint()
            sorted_indices = torch.argsort(rows)
            rows = rows[sorted_indices]   # [DH]
            cols = cols[sorted_indices]   # [DH]
            
            rows = torch.clamp(rows, 0, self.max_depth - 1)
            cols = torch.clamp(cols, 0, self.max_nnodes - 1)
            
            input_ids = self.input_ids_matrix[rows, cols]
            position_ids = rows + 1
            parents = self.parents_matrix[rows, cols]
            
            # source_idx parent_idx 
            # depth : depth d
            depth_offsets = torch.zeros(max(1, self.depth), dtype=torch.long, device=self.device)
            if self.depth > 1:
                depth_width_tensor = torch.tensor(self.depth_widths[:self.depth - 1], dtype=torch.long, device=self.device)
                depth_offsets[1:self.depth] = torch.cumsum(depth_width_tensor, dim=0)
            source_idx = depth_offsets[rows] + cols

            parent_rows = torch.clamp(rows - 1, min=0)
            parent_offsets = depth_offsets[parent_rows]
            parent_idx = torch.where(rows > 0, parent_offsets + parents, parents.to(dtype=torch.long))

            # parents_index : source_idx parent_idx
            n_selected = int(len(top_index))
            parents_index = torch.zeros(n_selected, dtype=torch.long, device=self.device)
            if n_selected > 0:
                sorted_source_idx, sorted_order = torch.sort(source_idx)
                search_pos = torch.searchsorted(sorted_source_idx, parent_idx)
                valid = (
                    (search_pos >= 0)
                    & (search_pos < n_selected)
                    & (sorted_source_idx[torch.clamp(search_pos, 0, n_selected - 1)] == parent_idx)
                )
                parents_index[valid] = sorted_order[search_pos[valid]]

                # fallback: depth 0 0, i-1
                if (~valid).any():
                    idx_arange = torch.arange(n_selected, device=self.device, dtype=torch.long)
                    fallback_prev = torch.clamp(idx_arange - 1, min=0)
                    fallback_vals = torch.where(rows == 0, torch.zeros_like(fallback_prev), fallback_prev)
                    parents_index[~valid] = fallback_vals[~valid]
            
            # parents_index
            parents_index = torch.clamp(parents_index, 0, len(top_index) - 1)
            
            # : parents_index top_index
            if len(parents_index) != len(top_index):
                raise ValueError(f"parents_index size mismatch: {len(parents_index)} != {len(top_index)}")
            
            attention_mask = self.generate_attention_mask(parents_index, len(top_index))
            
            outputs = {
                "input_ids": input_ids,  # [DH] 
                "position_ids": position_ids,  # [DH] depth
                "attention_mask": attention_mask,
                "parent_last": parents_index,  # [DH] 
                "is_final": True,
                "rows": rows,
                "cols": cols,
            }
            self.last_finalize_time_sec = max(0.0, float(time.time() - finalize_start_time))
        
        # tree
        self.prev_final_nnodes = self.final_nnodes
        self.prev_depth = self.depth
        self.prev_width = self.current_width
        self.prev_objective_value = objective_value
        
        return outputs
    
    def get_width_timing_stats(self):
        """Return model.model() call time statistics by width as a dictionary"""
        if not self.width_times:
            return {}
        
        stats = {}
        for width in sorted(self.width_times.keys()):
            times = self.width_times[width]
            count = len(times)
            total_time_ms = sum(times) * 1000
            avg_time_ms = (sum(times) / count) * 1000
            min_time_ms = min(times) * 1000
            max_time_ms = max(times) * 1000
            
            stats[width] = {
                'count': count,
                'total_time_ms': total_time_ms,
                'avg_time_ms': avg_time_ms,
                'min_time_ms': min_time_ms,
                'max_time_ms': max_time_ms
            }
        
        return stats
    
    def print_width_timing_stats(self):
        """Output average execution time of model.model() call time by width"""
        stats = self.get_width_timing_stats()
        if not stats:
            return
        
        print("\n" + "="*80)
        print("model.model() call timing statistics by width")
        print("="*80)
        print(f"{'Width':<10} {'Call Count':<12} {'Total Time (ms)':<15} {'Average Time (ms)':<15} {'Min Time (ms)':<15} {'Max Time (ms)':<15}")
        print("-"*80)
        
        for width in sorted(stats.keys()):
            s = stats[width]
            print(f"{width:<10} {s['count']:<12} {s['total_time_ms']:<15.3f} {s['avg_time_ms']:<15.3f} {s['min_time_ms']:<15.3f} {s['max_time_ms']:<15.3f}")
        
        print("="*80 + "\n")

    def print_tree_structure(self, tokenizer=None):
        """
        Function that prints the tree structure visually
        Args:
            tokenizer: tokenizer for converting token IDs to text (optional)
        """
        print("\n" + "="*80)
        print(f"Tree Structure (Current Depth: {self.depth}, Max Depth: {self.max_depth})")
        print("="*80)
        
        if self.depth == 0:
            print("Tree is empty (depth=0)")
            return
        
        # depth
        for d in range(self.depth):
            width = self.depth_widths[d] if d < len(self.depth_widths) else 0
            print(f"\nDepth {d} (width={width}):")
            print("-" * 80)
            
            for i in range(width):
                token_id = self.input_ids_matrix[d, i].item()
                weight = self.weight_matrix[d, i].item()
                parent_idx = self.parents_matrix[d, i].item() if d > 0 else -1
                
                if tokenizer:
                    try:
                        token_text = tokenizer.decode([token_id])
                        token_text = repr(token_text)
                    except:
                        token_text = "<?>"
                else:
                    token_text = ""
                
                if d == 0:
                    print(f"  [{d},{i:2d}] token_id={token_id:5d} {token_text:20s} weight={weight:.6f}")
                else:
                    print(f"  [{d},{i:2d}] token_id={token_id:5d} {token_text:20s} weight={weight:.6f} <- parent=[{d-1},{parent_idx:2d}]")
        
        print("="*80 + "\n")
    
    def print_tree_structure_hierarchical(self, tokenizer=None):
        """
        Function that prints the tree structure hierarchically (represented as a tree)
        Args:
            tokenizer: tokenizer for converting token IDs to text (optional)
        """
        print("\n" + "="*100)
        print(f"Tree Structure - Hierarchical View (Current Depth: {self.depth})")
        print("="*100)
        
        if self.depth == 0:
            print("Tree is empty (depth=0)")
            return
        
        # depth -
        children_map = {}  # {(depth, idx): [(child_depth, child_idx), ...]}
        
        for d in range(self.depth):
            width = self.depth_widths[d] if d < len(self.depth_widths) else 0
            for i in range(width):
                if d > 0:
                    parent_idx = self.parents_matrix[d, i].item()
                    parent_key = (d-1, parent_idx)
                    if parent_key not in children_map:
                        children_map[parent_key] = []
                    children_map[parent_key].append((d, i))
        
        def print_node(depth, idx, prefix="", is_last=True):
            width = self.depth_widths[depth] if depth < len(self.depth_widths) else 0
            if idx >= width:
                return
            
            token_id = self.input_ids_matrix[depth, idx].item()
            weight = self.weight_matrix[depth, idx].item()
            
            if tokenizer:
                try:
                    token_text = tokenizer.decode([token_id])
                    token_text = repr(token_text)[:15]
                except:
                    token_text = "<?>"
            else:
                token_text = f"id={token_id}"
            
            connector = "└─" if is_last else "├─"
            print(f"{prefix}{connector} [{depth},{idx:2d}] {token_text:15s} (w={weight:.4f})")
            
            node_key = (depth, idx)
            if node_key in children_map:
                children = children_map[node_key]
                extension = "   " if is_last else "│  "
                for i, child in enumerate(children):
                    is_last_child = (i == len(children) - 1)
                    print_node(child[0], child[1], prefix + extension, is_last_child)
        
        # Depth 0
        width_0 = self.depth_widths[0] if len(self.depth_widths) > 0 else 0
        for i in range(width_0):
            is_last = (i == width_0 - 1)
            print_node(0, i, "", is_last)
        
        print("="*100 + "\n")
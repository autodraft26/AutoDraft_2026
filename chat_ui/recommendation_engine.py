from typing import Dict, List, Optional, Tuple


def _is_dominated(a: Dict, b: Dict, metric_key: str = "metric_per_1m") -> bool:
    # a is dominated by b if b is no worse in both dimensions and better in at least one.
    am = a.get(metric_key)
    bm = b.get(metric_key)
    at = a.get("throughput_tps")
    bt = b.get("throughput_tps")
    if am is None or bm is None:
        return False
    if at is None or bt is None:
        return False
    no_worse = bm <= am and bt >= at
    strictly_better = bm < am or bt > at
    return bool(no_worse and strictly_better)


def pareto_optimal_rows(rows: List[Dict], metric_key: str = "metric_per_1m") -> List[Dict]:
    out = []
    for i, row in enumerate(rows):
        dominated = False
        for j, other in enumerate(rows):
            if i == j:
                continue
            if _is_dominated(row, other, metric_key=metric_key):
                dominated = True
                break
        if not dominated:
            out.append(row)
    return out


def _extract_metric(row: Dict, objective_mode: str, metric_preference: str) -> Optional[float]:
    # blend: prioritize api cost metric if available.
    # constraint: same metric key but supports future extension.
    if metric_preference == "total_cost":
        val = row.get("total_cost_per_1m")
        if val is None:
            val = row.get("cost_per_1m_tokens")
        if val is None:
            val = row.get("metric_per_1m")
        return float(val) if val is not None else None
    if metric_preference == "api_cost":
        val = row.get("api_cost_per_1m")
        if val is None:
            val = row.get("target_cost_per_1m_tokens")
        if val is None:
            val = row.get("metric_per_1m")
        return float(val) if val is not None else None
    if metric_preference == "draft_energy":
        val = row.get("draft_energy_per_1m_kwh")
        return float(val) if val is not None else None
    if metric_preference == "target_energy":
        val = row.get("target_energy_per_1m_kwh")
        return float(val) if val is not None else None
    val = row.get("metric_per_1m")
    return float(val) if val is not None else None


def build_recommendations(
    probe_rows: List[Dict], objective_mode: str, metric_preference: str
) -> Tuple[List[Dict], Dict]:
    scored = []
    for row in probe_rows:
        if not row.get("ok"):
            continue
        metric_value = _extract_metric(row, objective_mode, metric_preference)
        merged = dict(row)
        merged["selected_metric_per_1m"] = metric_value
        scored.append(merged)

    if not scored:
        return [], {
            "fastest": None,
            "best_efficiency": None,
            "pareto_optimal_ids": [],
        }

    fastest = max(
        scored, key=lambda r: float(r.get("throughput_tps") or 0.0)
    )
    best_eff_candidates = [r for r in scored if r.get("selected_metric_per_1m") is not None]
    best_eff = (
        min(
            best_eff_candidates,
            key=lambda r: (
                float(r.get("selected_metric_per_1m")),
                -float(r.get("throughput_tps") or 0.0),
            ),
        )
        if best_eff_candidates
        else None
    )

    pareto_input = []
    for row in scored:
        if row.get("selected_metric_per_1m") is None:
            continue
        p = dict(row)
        p["metric_per_1m"] = row.get("selected_metric_per_1m")
        pareto_input.append(p)
    pareto_rows = pareto_optimal_rows(pareto_input, metric_key="metric_per_1m")
    pareto_ids = [f'{r.get("server_id")}::{r.get("model_id")}' for r in pareto_rows]

    for row in scored:
        uid = f'{row.get("server_id")}::{row.get("model_id")}'
        tags = []
        if uid in pareto_ids:
            tags.append("pareto_optimal")
        if uid == f'{fastest.get("server_id")}::{fastest.get("model_id")}':
            tags.append("fastest")
        if best_eff and uid == f'{best_eff.get("server_id")}::{best_eff.get("model_id")}':
            tags.append("best_efficiency")
        row["recommendation_tags"] = tags
        row["recommended"] = bool(tags)
    return scored, {
        "fastest": {
            "server_id": fastest.get("server_id"),
            "model_id": fastest.get("model_id"),
        },
        "best_efficiency": (
            {
                "server_id": best_eff.get("server_id"),
                "model_id": best_eff.get("model_id"),
            }
            if best_eff
            else None
        ),
        "pareto_optimal_ids": pareto_ids,
    }

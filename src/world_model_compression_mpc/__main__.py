from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .asvd_compression import (
    compress_activation_aware_svd,
    compress_weight_svd,
    evaluate_output_mse,
)
from .cem_mpc_planner import CEMTrajectoryPlanner, PushTBlockEnv, TwoRoomDoorwayEnv
from .operator_distillation import CachedTrajectorySample, HybridOperatorDistiller


def run_benchmark(seed: int = 42) -> List[Dict[str, Any]]:
    """Run reproducible Activation-Aware SVD, TwoRoom/PushT CEM MPC, and Hybrid Operator Distillation benchmarks."""
    # 1. Activation-Aware SVD vs Weight SVD under skewed channel activations
    weight = [
        [0.45, 1.20, 0.05, 0.02],
        [0.40, -1.10, 0.04, -0.01],
        [-0.35, 0.95, -0.03, 0.02],
        [0.50, 1.05, 0.02, 0.01],
    ]
    calibration_inputs = [
        [25.0, 0.15, 0.10, 0.05],
        [-22.0, -0.10, 0.08, -0.04],
        [28.0, 0.12, -0.06, 0.03],
        [-24.0, 0.09, 0.05, 0.02],
    ]
    w_svd = compress_weight_svd(weight, rank=1)
    a_svd = compress_activation_aware_svd(weight, calibration_inputs, rank=1, alpha=0.8)
    mse_w = evaluate_output_mse(weight, w_svd, calibration_inputs)
    mse_a = evaluate_output_mse(weight, a_svd, calibration_inputs)

    # 2. TwoRoom doorway & PushT CEM trajectory planning
    two_room = TwoRoomDoorwayEnv(wall_x=0.0, door_half_width=0.25, dt=0.3)
    planner = CEMTrajectoryPlanner(
        horizon=5,
        action_dim=2,
        num_samples=28,
        num_elites=6,
        opt_iterations=4,
        seed=seed,
    )
    plan_res = planner.plan(initial_state=[-0.5, 0.0], goal_state=[0.6, 0.0], step_fn=two_room.step)

    pusht = PushTBlockEnv(dt=0.25)
    pusht_next = pusht.step([0.0, 0.0, 0.0], [0.8, 0.4, 0.2])

    # 3. Hybrid KL + Elite operator distillation
    cache = [
        CachedTrajectorySample(
            latent_feature=(1.0, 0.2, -0.5),
            teacher_action_mean=(0.8, 0.4),
            teacher_action_std=(0.25, 0.25),
            teacher_cost=0.15,
            is_elite=1,
        ),
        CachedTrajectorySample(
            latent_feature=(-1.0, -0.3, 0.6),
            teacher_action_mean=(-0.7, -0.5),
            teacher_action_std=(0.25, 0.25),
            teacher_cost=1.95,
            is_elite=0,
        ),
    ]
    distiller = HybridOperatorDistiller(feature_dim=3, action_dim=2, elite_loss_weight=1.5, learning_rate=0.2)
    init_loss = distiller.fit_epoch(cache, use_hybrid_elite_loss=True)
    final_loss = init_loss
    for _ in range(20):
        final_loss = distiller.fit_epoch(cache, use_hybrid_elite_loss=True)

    return [
        {
            "benchmark": "activation_aware_svd_rank1",
            "primary_metric": "asvd_output_mse",
            "score": round(mse_a, 4),
            "secondary_metric": "wsvd_output_mse",
            "secondary_value": round(mse_w, 4),
            "status": "ASVD_SUPERIOR" if mse_a < mse_w else "PARITY",
        },
        {
            "benchmark": "two_room_doorway_cem_mpc",
            "primary_metric": "best_terminal_cost",
            "score": round(plan_res.best_cost, 4),
            "secondary_metric": "pusht_dx_displacement",
            "secondary_value": round(pusht_next[0], 4),
            "status": "GOAL_REACHED" if plan_res.best_cost < 2.0 else "SUBOPTIMAL",
        },
        {
            "benchmark": "hybrid_operator_distillation",
            "primary_metric": "final_hybrid_total_loss",
            "score": round(final_loss.hybrid_total_loss, 4),
            "secondary_metric": "initial_hybrid_total_loss",
            "secondary_value": round(init_loss.hybrid_total_loss, 4),
            "status": "CONVERGED" if final_loss.hybrid_total_loss < init_loss.hybrid_total_loss else "DIVERGED",
        },
    ]


def format_markdown_table(rows: Sequence[Dict[str, Any]]) -> str:
    headers = [
        "benchmark",
        "primary_metric",
        "score",
        "secondary_metric",
        "secondary_value",
        "status",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def summarize_compression_tradeoff(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize benchmark pass count and ASVD vs WSVD MSE improvement ratio."""
    if not rows:
        return {"total_benchmarks": 0, "all_converged": False, "asvd_mse_ratio": 1.0}
    asvd_row = next((r for r in rows if r.get("benchmark") == "activation_aware_svd_rank1"), None)
    ratio = 1.0
    if asvd_row and float(asvd_row.get("secondary_value", 0.0)) > 0:
        ratio = round(float(asvd_row["score"]) / float(asvd_row["secondary_value"]), 4)
    passing = sum(
        1 for r in rows if r.get("status") in ("ASVD_SUPERIOR", "GOAL_REACHED", "CONVERGED")
    )
    return {
        "total_benchmarks": len(rows),
        "passing_benchmarks": passing,
        "all_converged": passing == len(rows),
        "asvd_mse_ratio": ratio,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run reproducible world-model-compression-mpc ASVD, CEM MPC, and distillation benchmarks."
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format for benchmark results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible benchmark execution.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional file path to write benchmark output.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    rows = run_benchmark(seed=args.seed)
    rendered = (
        json.dumps({"seed": args.seed, "benchmarks": rows}, indent=2)
        if args.format == "json"
        else format_markdown_table(rows)
    )
    if args.output is not None:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"Error writing output to {args.output}: {exc}", file=sys.stderr)
            return 1
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

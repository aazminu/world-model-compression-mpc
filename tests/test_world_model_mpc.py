from __future__ import annotations

import pytest

from world_model_compression_mpc import (
    CachedTrajectorySample,
    CEMTrajectoryPlanner,
    HybridOperatorDistiller,
    PushTBlockEnv,
    TwoRoomDoorwayEnv,
    compress_activation_aware_svd,
    compress_weight_svd,
    evaluate_output_mse,
    rank_for_param_budget,
)


def test_asvd_beats_weight_svd_under_skewed_channel_activations() -> None:
    # 4x4 weight matrix where column 0 has smaller weight norm than column 1,
    # but input channel 0 has 100x larger activation amplitude than channel 1.
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
    assert mse_a < mse_w
    assert rank_for_param_budget(8, 8, 0.5) == 2

    # Repeated/degenerate singular values (4x4 identity matrix) must reconstruct exactly at full rank
    ident = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    ident_recon = compress_weight_svd(ident, rank=4).reconstruct_matrix()
    for i in range(4):
        for j in range(4):
            assert ident_recon[i][j] == pytest.approx(ident[i][j], abs=1e-9)



def test_two_room_and_pusht_cem_planning() -> None:
    two_room = TwoRoomDoorwayEnv(wall_x=0.0, door_half_width=0.25, dt=0.3)
    # Trying to cross wall outside doorway gap (y=0.8) gets blocked
    blocked = two_room.step([-0.1, 0.8], [1.0, 0.0])
    assert blocked[0] < 0.0
    # Crossing through doorway gap (y=0.0) succeeds
    through = two_room.step([-0.1, 0.0], [1.0, 0.0])
    assert through[0] > 0.0

    planner = CEMTrajectoryPlanner(horizon=5, action_dim=2, num_samples=28, num_elites=6, opt_iterations=4, seed=7)
    plan_res = planner.plan(initial_state=[-0.5, 0.0], goal_state=[0.6, 0.0], step_fn=two_room.step)
    assert len(plan_res.best_action_sequence) == 5
    assert len(plan_res.elite_indices) == 6
    assert plan_res.best_cost < 2.0

    pusht = PushTBlockEnv(dt=0.25)
    next_pose = pusht.step([0.0, 0.0, 0.0], [0.8, 0.4, 0.2])
    assert next_pose[0] > 0.0 and next_pose[1] > 0.0


def test_hybrid_operator_distillation_improves_kl_and_elite_classification() -> None:
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
    initial_losses = distiller.fit_epoch(cache, use_hybrid_elite_loss=True)
    for _ in range(20):
        final_losses = distiller.fit_epoch(cache, use_hybrid_elite_loss=True)

    assert final_losses.forward_kl_loss < initial_losses.forward_kl_loss
    assert final_losses.elite_bce_loss < initial_losses.elite_bce_loss
    assert distiller.predict_elite_prob(cache[0].latent_feature) > distiller.predict_elite_prob(cache[1].latent_feature)


def test_cli_benchmark_markdown_json_and_reproducibility(
    capsys: pytest.CaptureFixture[str], tmp_path: pytest.TempPathFactory
) -> None:
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path
    from world_model_compression_mpc.__main__ import format_markdown_table, main, run_benchmark

    rows_1 = run_benchmark(seed=42)
    rows_2 = run_benchmark(seed=42)
    rows_3 = run_benchmark(seed=99)
    assert rows_1 == rows_2
    assert rows_1 != rows_3
    assert len(rows_1) == 3

    md = format_markdown_table(rows_1)
    assert "| benchmark | primary_metric | score |" in md
    assert "activation_aware_svd_rank1" in md
    assert "two_room_doorway_cem_mpc" in md

    out_path = Path(str(tmp_path)) / "mpc_bench.json"
    rc = main(["--format", "json", "--seed", "42", "--output", str(out_path)])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["seed"] == 42
    assert data["benchmarks"] == rows_1
    assert json.loads(out_path.read_text(encoding="utf-8")) == data

    err_rc = main(["--format", "markdown", "--output", str(tmp_path)])
    assert err_rc == 1
    assert "Error writing output" in capsys.readouterr().err

    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "benchmark.py"), "--format", "markdown"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "hybrid_operator_distillation" in proc.stdout

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    proc_mod = subprocess.run(
        [sys.executable, "-m", "world_model_compression_mpc", "--format", "json", "--seed", "42"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert json.loads(proc_mod.stdout)["seed"] == 42

    with pytest.raises(SystemExit):
        main(["--format", "invalid_fmt"])



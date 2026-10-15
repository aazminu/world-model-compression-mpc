from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

Matrix = List[List[float]]
Vector = List[float]


@dataclass(frozen=True)
class LowRankLinearLayer:
    """Factored linear layer W_approx = left_factor @ right_factor with rank r."""

    left_factor: Matrix   # shape: (d_out, r)
    right_factor: Matrix  # shape: (r, d_in)
    rank: int
    retained_param_ratio: float

    def forward(self, x: Sequence[float]) -> Vector:
        # Compute z = right_factor @ x (shape r), then y = left_factor @ z (shape d_out)
        z = [
            sum(row[j] * x[j] for j in range(len(x)))
            for row in self.right_factor
        ]
        return [
            sum(row[k] * z[k] for k in range(self.rank))
            for row in self.left_factor
        ]

    def reconstruct_matrix(self) -> Matrix:
        d_out = len(self.left_factor)
        d_in = len(self.right_factor[0]) if self.right_factor else 0
        out: Matrix = []
        for i in range(d_out):
            row: Vector = []
            for j in range(d_in):
                val = sum(self.left_factor[i][k] * self.right_factor[k][j] for k in range(self.rank))
                row.append(val)
            out.append(row)
        return out


def _mat_vec(mat: Matrix, vec: Sequence[float]) -> Vector:
    return [sum(r[j] * vec[j] for j in range(len(vec))) for r in mat]


def _transpose(mat: Matrix) -> Matrix:
    if not mat:
        return []
    return [[mat[i][j] for i in range(len(mat))] for j in range(len(mat[0]))]


def _truncated_svd_power(mat: Matrix, rank: int, iters: int = 60) -> Tuple[Matrix, Vector, Matrix]:
    """Computes top-r singular triplets (U_r, S_r, V_r^T) via deflated power iteration."""
    d_out = len(mat)
    d_in = len(mat[0])
    residual = [list(row) for row in mat]
    u_cols: List[Vector] = []
    s_vals: Vector = []
    vt_rows: List[Vector] = []

    for r_idx in range(rank):
        # Deterministic non-zero initial vector
        v = [math.cos(0.7 * (j + 1) + 1.3 * (r_idx + 1)) for j in range(d_in)]
        v_norm = math.sqrt(sum(x * x for x in v)) or 1.0
        v = [x / v_norm for x in v]

        for _ in range(iters):
            u = _mat_vec(residual, v)
            u_norm = math.sqrt(sum(x * x for x in u))
            if u_norm < 1e-12:
                break
            u = [x / u_norm for x in u]
            res_t = _transpose(residual)
            v_next = _mat_vec(res_t, u)
            # Orthogonalize against previously found right singular vectors
            for prev_v in vt_rows:
                proj = sum(v_next[j] * prev_v[j] for j in range(d_in))
                v_next = [v_next[j] - proj * prev_v[j] for j in range(d_in)]
            v_norm = math.sqrt(sum(x * x for x in v_next))
            if v_norm < 1e-12:
                break
            v = [x / v_norm for x in v_next]

        av = _mat_vec(residual, v)
        sigma = math.sqrt(sum(x * x for x in av))
        if sigma < 1e-12:
            u = [1.0 if i == (r_idx % d_out) else 0.0 for i in range(d_out)]
            sigma = 0.0
        else:
            u = [x / sigma for x in av]

        u_cols.append(u)
        s_vals.append(sigma)
        vt_rows.append(v)

        for i in range(d_out):
            for j in range(d_in):
                residual[i][j] -= sigma * u[i] * v[j]

    u_mat = [[u_cols[k][i] for k in range(rank)] for i in range(d_out)]
    return u_mat, s_vals, vt_rows


def rank_for_param_budget(d_out: int, d_in: int, target_retention_ratio: float) -> int:
    if not (0.05 <= target_retention_ratio <= 1.0):
        raise ValueError("target_retention_ratio must be in [0.05, 1.0]")
    raw_rank = int(round((target_retention_ratio * d_out * d_in) / (d_out + d_in)))
    return max(1, min(min(d_out, d_in), raw_rank))


def compress_weight_svd(weight: Matrix, rank: int) -> LowRankLinearLayer:
    d_out = len(weight)
    d_in = len(weight[0])
    if not (1 <= rank <= min(d_out, d_in)):
        raise ValueError("rank out of valid bounds")
    u_mat, s_vals, vt_rows = _truncated_svd_power(weight, rank)
    left = [[u_mat[i][k] * math.sqrt(s_vals[k]) for k in range(rank)] for i in range(d_out)]
    right = [[math.sqrt(s_vals[k]) * vt_rows[k][j] for j in range(d_in)] for k in range(rank)]
    ratio = (rank * (d_out + d_in)) / float(d_out * d_in)
    return LowRankLinearLayer(left_factor=left, right_factor=right, rank=rank, retained_param_ratio=round(ratio, 4))


def compress_activation_aware_svd(
    weight: Matrix,
    calibration_inputs: Sequence[Sequence[float]],
    rank: int,
    alpha: float = 0.5,
) -> LowRankLinearLayer:
    """Activation-Aware SVD (ASVD): scales weight columns by channel RMS^alpha prior to SVD."""
    if not calibration_inputs:
        raise ValueError("calibration_inputs cannot be empty")
    d_out = len(weight)
    d_in = len(weight[0])
    if not (1 <= rank <= min(d_out, d_in)):
        raise ValueError("rank out of valid bounds")

    scales: Vector = []
    n_cal = len(calibration_inputs)
    for j in range(d_in):
        mean_sq = sum(x[j] * x[j] for x in calibration_inputs) / n_cal
        rms = math.sqrt(max(mean_sq, 1e-8))
        scales.append(rms ** alpha)

    scaled_w = [[weight[i][j] * scales[j] for j in range(d_in)] for i in range(d_out)]
    u_mat, s_vals, vt_rows = _truncated_svd_power(scaled_w, rank)

    left = [[u_mat[i][k] * math.sqrt(s_vals[k]) for k in range(rank)] for i in range(d_out)]
    right = [
        [(math.sqrt(s_vals[k]) * vt_rows[k][j]) / scales[j] for j in range(d_in)]
        for k in range(rank)
    ]
    ratio = (rank * (d_out + d_in)) / float(d_out * d_in)
    return LowRankLinearLayer(left_factor=left, right_factor=right, rank=rank, retained_param_ratio=round(ratio, 4))


def evaluate_output_mse(
    teacher_weight: Matrix,
    compressed_layer: LowRankLinearLayer,
    eval_inputs: Sequence[Sequence[float]],
) -> float:
    if not eval_inputs:
        raise ValueError("eval_inputs cannot be empty")
    total_sq = 0.0
    d_out = len(teacher_weight)
    for x in eval_inputs:
        y_true = _mat_vec(teacher_weight, x)
        y_hat = compressed_layer.forward(x)
        total_sq += sum((a - b) ** 2 for a, b in zip(y_true, y_hat)) / d_out
    return total_sq / len(eval_inputs)

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class CachedTrajectorySample:
    latent_feature: Tuple[float, ...]
    teacher_action_mean: Tuple[float, ...]
    teacher_action_std: Tuple[float, ...]
    teacher_cost: float
    is_elite: int  # 1 if in top-K teacher CEM elites, else 0


@dataclass(frozen=True)
class DistillationLosses:
    forward_kl_loss: float
    elite_bce_loss: float
    hybrid_total_loss: float


def gaussian_forward_kl(
    teacher_mean: Sequence[float],
    teacher_std: Sequence[float],
    student_mean: Sequence[float],
    student_std: Sequence[float],
) -> float:
    """Diagonal Gaussian Forward KL: D_KL(N(mu_T, sigma_T^2) || N(mu_S, sigma_S^2))."""
    kl = 0.0
    for mt, st, ms, ss in zip(teacher_mean, teacher_std, student_mean, student_std):
        st_safe = max(st, 1e-4)
        ss_safe = max(ss, 1e-4)
        kl += (
            math.log(ss_safe / st_safe)
            + (st_safe * st_safe + (mt - ms) ** 2) / (2.0 * ss_safe * ss_safe)
            - 0.5
        )
    return kl


class HybridOperatorDistiller:
    """Trains a compressed student scoring head with Forward KL + Elite-Set Binary Cross-Entropy."""

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        elite_loss_weight: float = 1.5,
        learning_rate: float = 0.15,
    ) -> None:
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.elite_loss_weight = elite_loss_weight
        self.learning_rate = learning_rate
        self.action_weights: List[List[float]] = [
            [0.0 for _ in range(feature_dim)] for _ in range(action_dim)
        ]
        self.elite_weights: List[float] = [0.0 for _ in range(feature_dim)]
        self.elite_bias: float = 0.0

    @staticmethod
    def _sigmoid(z: float) -> float:
        if z >= 0.0:
            return 1.0 / (1.0 + math.exp(-min(z, 50.0)))
        ez = math.exp(max(z, -50.0))
        return ez / (1.0 + ez)

    def predict_action_mean(self, feat: Sequence[float]) -> List[float]:
        return [
            sum(self.action_weights[d][j] * feat[j] for j in range(self.feature_dim))
            for d in range(self.action_dim)
        ]

    def predict_elite_prob(self, feat: Sequence[float]) -> float:
        z = self.elite_bias + sum(self.elite_weights[j] * feat[j] for j in range(self.feature_dim))
        return self._sigmoid(z)

    def fit_epoch(
        self,
        cache: Sequence[CachedTrajectorySample],
        use_hybrid_elite_loss: bool = True,
    ) -> DistillationLosses:
        if not cache:
            raise ValueError("cache cannot be empty")
        total_kl = 0.0
        total_bce = 0.0
        n = len(cache)
        student_std = [0.35] * self.action_dim

        for sample in cache:
            pred_mean = self.predict_action_mean(sample.latent_feature)
            kl = gaussian_forward_kl(
                sample.teacher_action_mean,
                sample.teacher_action_std,
                pred_mean,
                student_std,
            )
            total_kl += kl

            # Gradient of Forward KL w.r.t student_mean_d is (student_mean_d - teacher_mean_d) / sigma_S^2
            for d in range(self.action_dim):
                grad_m = (pred_mean[d] - sample.teacher_action_mean[d]) / (student_std[d] ** 2)
                for j in range(self.feature_dim):
                    self.action_weights[d][j] -= (self.learning_rate / n) * grad_m * sample.latent_feature[j]

            p_elite = self.predict_elite_prob(sample.latent_feature)
            y = float(sample.is_elite)
            p_clamped = max(1e-6, min(1.0 - 1e-6, p_elite))
            bce = -(y * math.log(p_clamped) + (1.0 - y) * math.log(1.0 - p_clamped))
            total_bce += bce

            if use_hybrid_elite_loss:
                err = (p_elite - y) * self.elite_loss_weight
                self.elite_bias -= (self.learning_rate / n) * err
                for j in range(self.feature_dim):
                    self.elite_weights[j] -= (self.learning_rate / n) * err * sample.latent_feature[j]

        mean_kl = total_kl / n
        mean_bce = total_bce / n
        hybrid = mean_kl + (self.elite_loss_weight * mean_bce if use_hybrid_elite_loss else 0.0)
        return DistillationLosses(
            forward_kl_loss=round(mean_kl, 5),
            elite_bce_loss=round(mean_bce, 5),
            hybrid_total_loss=round(hybrid, 5),
        )

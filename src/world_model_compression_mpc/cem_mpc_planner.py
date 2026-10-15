from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

Vector = List[float]


@dataclass(frozen=True)
class CEMPlanResult:
    best_action_sequence: List[Vector]
    mean_action_sequence: List[Vector]
    std_action_sequence: List[Vector]
    elite_indices: List[int]
    best_cost: float


class TwoRoomDoorwayEnv:
    """2D continuous navigation environment with a wall at x=0 and a narrow doorway gap at y in [-0.25, 0.25]."""

    def __init__(self, wall_x: float = 0.0, door_half_width: float = 0.25, dt: float = 0.25) -> None:
        self.wall_x = wall_x
        self.door_half_width = door_half_width
        self.dt = dt

    def step(self, state: Sequence[float], action: Sequence[float]) -> Vector:
        x0, y0 = state[0], state[1]
        vx = max(-1.5, min(1.5, action[0]))
        vy = max(-1.5, min(1.5, action[1]))
        x1 = x0 + self.dt * vx
        y1 = y0 + self.dt * vy

        # Check wall crossing at x = wall_x
        crossed_wall = (x0 < self.wall_x <= x1) or (x1 <= self.wall_x < x0)
        if crossed_wall and abs(0.5 * (y0 + y1)) > self.door_half_width:
            # Blocked by wall outside doorway aperture
            x1 = self.wall_x - 0.02 if x0 < self.wall_x else self.wall_x + 0.02
        return [round(x1, 5), round(y1, 5)]


class PushTBlockEnv:
    """Planar T-block manipulation dynamics (block_x, block_y, block_theta) pushed by end-effector delta."""

    def __init__(self, dt: float = 0.2) -> None:
        self.dt = dt

    def step(self, state: Sequence[float], action: Sequence[float]) -> Vector:
        bx, by, theta = state[0], state[1], state[2]
        ux = max(-1.0, min(1.0, action[0]))
        uy = max(-1.0, min(1.0, action[1]))
        utheta = max(-1.0, min(1.0, action[2])) if len(action) > 2 else 0.35 * (ux - uy)
        return [
            round(bx + self.dt * ux, 5),
            round(by + self.dt * uy, 5),
            round(theta + self.dt * utheta, 5),
        ]


class CEMTrajectoryPlanner:
    """Cross-Entropy Method (CEM) MPC planner over a learned or analytical transition dynamics model."""

    def __init__(
        self,
        horizon: int = 6,
        action_dim: int = 2,
        num_samples: int = 32,
        num_elites: int = 6,
        opt_iterations: int = 4,
        seed: int = 42,
    ) -> None:
        if num_elites < 1 or num_elites > num_samples:
            raise ValueError("num_elites must be between 1 and num_samples")
        self.horizon = horizon
        self.action_dim = action_dim
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.opt_iterations = opt_iterations
        self.seed = seed

    def plan(
        self,
        initial_state: Sequence[float],
        goal_state: Sequence[float],
        step_fn: Callable[[Sequence[float], Sequence[float]], Vector],
    ) -> CEMPlanResult:
        rng = random.Random(self.seed)
        means = [[0.0 for _ in range(self.action_dim)] for _ in range(self.horizon)]
        stds = [[1.0 for _ in range(self.action_dim)] for _ in range(self.horizon)]

        best_seq = [list(m) for m in means]
        best_cost = float("inf")
        last_elite_indices: List[int] = []

        for _ in range(self.opt_iterations):
            candidates: List[List[Vector]] = []
            costs: List[Tuple[float, int]] = []

            for s_idx in range(self.num_samples):
                seq: List[Vector] = []
                state = list(initial_state)
                rollout_cost = 0.0
                for t in range(self.horizon):
                    act = [
                        max(-1.5, min(1.5, rng.gauss(means[t][d], stds[t][d])))
                        for d in range(self.action_dim)
                    ]
                    seq.append(act)
                    state = step_fn(state, act)
                    dist_sq = sum((state[i] - goal_state[i]) ** 2 for i in range(len(goal_state)))
                    weight = 2.5 if t == self.horizon - 1 else 1.0
                    rollout_cost += weight * dist_sq

                candidates.append(seq)
                costs.append((rollout_cost, s_idx))

            costs.sort(key=lambda item: item[0])
            last_elite_indices = [idx for _, idx in costs[: self.num_elites]]
            if costs[0][0] < best_cost:
                best_cost = costs[0][0]
                best_seq = candidates[costs[0][1]]

            for t in range(self.horizon):
                for d in range(self.action_dim):
                    elite_vals = [candidates[idx][t][d] for idx in last_elite_indices]
                    m = sum(elite_vals) / self.num_elites
                    var = sum((v - m) ** 2 for v in elite_vals) / max(self.num_elites - 1, 1)
                    means[t][d] = round(m, 5)
                    stds[t][d] = round(max(math.sqrt(var), 0.05), 5)

        return CEMPlanResult(
            best_action_sequence=best_seq,
            mean_action_sequence=means,
            std_action_sequence=stds,
            elite_indices=last_elite_indices,
            best_cost=round(best_cost, 5),
        )

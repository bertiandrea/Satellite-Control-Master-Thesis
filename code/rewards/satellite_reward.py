# satellite_reward.py

from code.utils.satellite_util import quat_diff_rad

import isaacgym #BugFix
import torch

from abc import ABC, abstractmethod
import math
from typing import Optional

class RewardFunction(ABC):

    @abstractmethod
    def compute(self, quats: torch.Tensor, ang_vels: torch.Tensor, ang_accs: torch.Tensor, goal_quat: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    def _assert_valid_tensor(tensor: torch.Tensor, name: str):
        assert not torch.isnan(tensor).any(), f"{name} has NaN values"
        assert not torch.isinf(tensor).any(), f"{name} has Inf values"

class ExponentialReward(RewardFunction):
    def __init__(self, lambda_u, goal_deg, bonus):
        self.prev_phi: Optional[torch.Tensor] = None
        self.lambda_u = lambda_u
        self.goal_rad = math.radians(goal_deg)
        self.bonus = bonus
        print(f"Initialized ExponentialReward with lambda_u={lambda_u}, goal_deg={goal_deg}, bonus={bonus}")

    def compute(self, quats, ang_vels, ang_accs, goal_quat, actions):
        phi = quat_diff_rad(quats, goal_quat)
        r_q = torch.exp(torch.div(torch.neg(phi), 0.14 * 2 * math.pi))

        if self.prev_phi is None:
            reward = r_q
        else:
            reward = torch.where(
                torch.gt(torch.sub(phi, self.prev_phi), 0.0),
                torch.sub(r_q, 1.0), r_q
            )

        self.prev_phi = phi.detach().clone()

        u_sq = torch.sum(torch.square(actions), dim=-1)
        r_effort = torch.mul(self.lambda_u, u_sq)

        bonus = torch.mul(
            torch.le(phi, self.goal_rad).to(phi.dtype),
            self.bonus
        )

        final_reward = torch.add(torch.sub(reward, r_effort), bonus)
        self._assert_valid_tensor(final_reward, "reward")

        return final_reward


REWARD_MAP = {
    "ExponentialReward": ExponentialReward,
}
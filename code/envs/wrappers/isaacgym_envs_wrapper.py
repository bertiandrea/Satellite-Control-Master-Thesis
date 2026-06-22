# isaacgym_envs.py

from code.envs.wrappers.base import Wrapper

import isaacgym #BugFix
import torch

from skrl.utils.spaces.torch import (
    convert_gym_space,
    flatten_tensorized_space,
    tensorize_space,
    unflatten_tensorized_space,
)

import gymnasium
from typing import Any, Tuple, Union

class IsaacGymWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        super().__init__(env)

        self._reset_once = True
        self._observations = None
        self._states = None
        self._info = {}

    @property
    def observation_space(self) -> gymnasium.Space:
        return convert_gym_space(self._unwrapped.observation_space)

    @property
    def action_space(self) -> gymnasium.Space:
        return convert_gym_space(self._unwrapped.action_space)

    @property
    def state_space(self) -> Union[gymnasium.Space, None]:
        return convert_gym_space(self._unwrapped.state_space)
        
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        obs_states_dict, reward, terminated, self._info = self._env.step(
            unflatten_tensorized_space(self.action_space, actions)
        )

        self._observations = flatten_tensorized_space(tensorize_space(self.observation_space, obs_states_dict["obs"]))
        self._states = flatten_tensorized_space(tensorize_space(self.state_space, obs_states_dict["states"]))

        truncated = self._info["time_outs"] if "time_outs" in self._info else torch.zeros_like(terminated)
        return self._states, reward.view(-1, 1), terminated.view(-1, 1), truncated.view(-1, 1), self._info

    def reset(self) -> Tuple[torch.Tensor, Any]:
        if self._reset_once:
            obs_states_dict = self._env.reset()

            self._observations = flatten_tensorized_space(tensorize_space(self.observation_space, obs_states_dict["obs"]))
            self._states = flatten_tensorized_space(tensorize_space(self.state_space, obs_states_dict["states"]))

            self._reset_once = False
        return self._states, self._info

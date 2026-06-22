# base.py

import isaacgym #BugFix
import torch

from skrl import config

from typing import Any, Union
import gymnasium

class Wrapper(object):
    def __init__(self, env: Any) -> None:
        self._env = env
        try:
            self._unwrapped = self._env.unwrapped
        except:
            self._unwrapped = env

        if hasattr(self._unwrapped, "device"):
            self._device = config.torch.parse_device(self._unwrapped.device)
        else:
            self._device = config.torch.parse_device(None)

    def __getattr__(self, key: str) -> Any:
        if hasattr(self._env, key):
            return getattr(self._env, key)
        if hasattr(self._unwrapped, key):
            return getattr(self._unwrapped, key)
        raise AttributeError(
            f"Wrapped environment ({self._unwrapped.__class__.__name__}) does not have attribute '{key}'"
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def num_envs(self) -> int:
        return self._unwrapped.num_envs

    @property
    def num_agents(self) -> int:
        return self._unwrapped.num_agents

    @property
    def state_space(self) -> Union[gymnasium.Space, None]:
        return self._unwrapped.state_space

    @property
    def observation_space(self) -> gymnasium.Space:
        return self._unwrapped.observation_space

    @property
    def action_space(self) -> gymnasium.Space:
        return self._unwrapped.action_space

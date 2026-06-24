# satellite_config.py

from pathlib import Path
import numpy as np

import isaacgym
import torch

from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL

HEADLESS = True
DEBUG_ARROWS = False
LOG_TRAJECTORIES = True

NUM_ENVS = 4096
EPISODE_LENGTH = 600.0

DR_RANDOMIZATION = False
EXPLOSION = False

CONFIG = {
    # --- devices ----------------------------------------------------
    "profile": False,
	
    "headless": HEADLESS,
	
    # --- env section -------------------------------------------------------
    "env": {
        "numEnvs": NUM_ENVS,
		
        "max_episode_length": EPISODE_LENGTH,
        
        "debug_arrows": DEBUG_ARROWS,
        "debug_prints": False,
        
        "discretize_starting_pos": True,
		
        "asset": {
            "asset_root": str(Path(__file__).resolve().parent.parent),
            "asset_file_name": "satellite.urdf",
            "asset_name": "satellite",
        },
    },

    # --- RL / PPO hyper-params --------------------------------------------
    "rl": {
        "PPO": {
            "num_envs": NUM_ENVS,
            
            "learning_rate_scheduler" : KLAdaptiveRL, # NOT SERIALIZABLE
            "state_preprocessor" : RunningStandardScaler, # NOT SERIALIZABLE
            "value_preprocessor" : RunningStandardScaler, # NOT SERIALIZABLE
            "rewards_shaper" : lambda rewards, timestep, timesteps: rewards * 0.01, # NOT SERIALIZABLE

            "experiment": {
                "write_interval": "auto",
                "checkpoint_interval": "auto",
                "directory": "./eval/runs",
                "wandb": False,
            }
        },
        "trainer": {
            "timesteps": int(EPISODE_LENGTH / ( 1.0 / 60.0 )),
            "disable_progressbar": False,
            "headless": HEADLESS,
            "stochastic_evaluation": False,
        },
    },

    # --- log_status -----------------------------------------------------------
    "log_status": {
        "log": True,
        "log_interval": 100,  # steps
        "log_dir": "./eval/status",
    },

    # --- log_trajectories -----------------------------------------------------------
    "log_trajectories": {
        "log": True,
        "log_interval": 100,  # steps
        "log_flush": 1000,    # steps
        "log_dir": "./eval/trajectories",
    },

    # --- explosion ---------------------------------------------------------
    "explosion": {
        "enabled": EXPLOSION,
        "explosion_time": 60,  # seconds
        "explosion_mean": 10.0,
        "explosion_std": 1.0,
    },

    # --- dr_randomization -------------------------------------------------
    "dr_randomization": {
        "enabled": DR_RANDOMIZATION,
    },
}
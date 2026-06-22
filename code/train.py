# train.py

from code.configs.satellite_config_train import CONFIG
from code.envs.satellite import Satellite
from code.models.custom_model import Shared
from code.envs.wrappers.isaacgym_envs_wrapper import IsaacGymWrapper

import isaacgym #BugFix
import torch

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

import json
import datetime
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Profiler imports
from torch.profiler import (
    profile,
    ProfilerActivity,
    tensorboard_trace_handler,
)
import os
import pandas as pd
# ──────────────────────────────────────────────────────────────────────────────

def setup_profiler(log_dir = "/home/andreaberti"):
    dir_path = log_dir + "/profiler_logs"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    #################################################################################
    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=tensorboard_trace_handler(dir_path),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        with_modules=True,
    )
    return prof

def save_profiler_results(prof, log_dir="/home/andreaberti"):
    events = prof.key_averages()
    #################################################################################
    output_path = log_dir + "/profiler_text/text_output.txt"
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(events.table(sort_by="self_cuda_time_total", row_limit=500))
        f.write("\n\n\n")
        f.write(events.table(sort_by="self_cpu_time_total", row_limit=500))
        f.write("\n\n\n")
        f.write(events.table(sort_by="self_cuda_memory_usage", row_limit=500))
        f.write("\n\n\n")
        f.write(events.table(sort_by="self_cpu_memory_usage", row_limit=500))
    #################################################################################
    rows = []
    for e in events:
        rows.append({
            "name":               e.key[:50],  # Truncate to 50 characters
            "self_cpu_time_ms":   e.self_cpu_time_total / 1e3,
            "cpu_time_ms":        e.cpu_time_total / 1e3,
            "self_cuda_time_ms":  e.self_device_time_total / 1e3,
            "cuda_time_ms":       e.device_time_total / 1e3,
            "self_cpu_memory_bytes":   e.self_cpu_memory_usage,
            "self_cuda_memory_bytes":  e.self_device_memory_usage,
            "cpu_memory_bytes":   e.cpu_memory_usage,
            "cuda_memory_bytes":  e.device_memory_usage,
            "count":              e.count,
            "flops":              e.flops,
            "device_type":        str(e.device_type),
        })
    df = pd.DataFrame(rows)
    df['order'] = df['name'].str[0].map({'#': 0, '$': 1}).fillna(2).astype(int)
    df = df.sort_values(['order', 'name'], ascending=[True, True])
    df = df.drop(columns='order')
    #################################################################################
    print(df.head(40))
    #################################################################################
    csv_path = log_dir + "/profiler_text/csv_output.csv"
    if not os.path.exists(os.path.dirname(csv_path)):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)

def main():
    if CONFIG["set_seed"]:
        set_seed(CONFIG["seed"])
    else:
        CONFIG["seed"] = torch.seed() % (2**32)
        set_seed(CONFIG["seed"])
    
    #################################################################################
    
    BASE_DIR = Path(__file__).resolve().parent.parent
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    CONFIG["rl"]["PPO"]["experiment"]["experiment_name"] = f"run_{timestamp}"
    CONFIG["log_status"]["log_dir"] = CONFIG["log_status"]["log_dir"] + f"/status_{timestamp}"

    # ──────────────────────────────────────────────────────────────────────────────
    run_dir = BASE_DIR / "train" / "configs"
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / f"config_{timestamp}.json"

    with open(config_path, "w") as f:
        json.dump(CONFIG, f, indent=4, default=str)

    print(f"[INFO] Config salvata in: {config_path}")
    # ──────────────────────────────────────────────────────────────────────────────

    print(CONFIG)
    
    env = Satellite(
        config=CONFIG,
        rl_device=CONFIG["rl_device"],
        sim_device=CONFIG["sim_device"],
        graphics_device_id=CONFIG["graphics_device_id"],
        headless=CONFIG["headless"],
        is_eval=False
    )
    
    env = IsaacGymWrapper(env)

    memory = RandomMemory(memory_size=CONFIG["rl"]["memory"]["rollouts"], num_envs=env.num_envs, device=env.device)

    models = {}
    models["policy"] = Shared(env.state_space, env.action_space, env.device)
    models["value"] = models["policy"]  # Shared model for policy and value
   
    CONFIG["rl"]["PPO"]["state_preprocessor_kwargs"] = {
        "size": env.state_space, "device": env.device
    }
    CONFIG["rl"]["PPO"]["value_preprocessor_kwargs"] = {
        "size": 1, "device": env.device
    }
    
    cfg_ppo = PPO_DEFAULT_CONFIG.copy()
    cfg_ppo.update(CONFIG["rl"]["PPO"])

    agent = PPO(models=models,
            memory=memory,
            cfg=cfg_ppo,
            observation_space=env.state_space,
            action_space=env.action_space,
            device=env.device)
    
    trainer = SequentialTrainer(cfg=CONFIG["rl"]["trainer"], env=env, agents=agent)

    if CONFIG["profile"]:
        prof = setup_profiler()
        prof.start()

    trainer.train()
    
    if CONFIG["profile"]:
        prof.stop()
        save_profiler_results(prof)
    
    #################################################################################

if __name__ == "__main__":
    main()
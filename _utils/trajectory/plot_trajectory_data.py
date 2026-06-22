import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import random
import os
import gc

# Impostazioni generali
N_ENV_PLOT = 8
HZ = 60
MAX_TIME = 200.0
SMOOTH_WINDOW = 20
LOG_PATH = "/home/andreaberti/Satellite-Control-Thesis-Baseline/eval/trajectories/randomization/0_05/nominal/run_20260417_164304/trajectories_20260615_130115.pt"
#LOG_PATH = "/home/andreaberti/Satellite-Control-Thesis-Baseline/eval/trajectories/trajectories_20260611_091501.pt"
CACHE_DIR = "/home/andreaberti/Satellite-Control-Thesis-Baseline/eval/trajectories/_cache_mmap"

def build_cache_with_low_ram():
    print("Inizio conversione del file .pt in formato binario (Memory-Mapped)...")
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    first_load = torch.load(LOG_PATH, map_location="cpu", weights_only=True)
    num_entries = len(first_load)
    num_envs = first_load[0]["quat"].shape[0]
    print(f"Rilevate {num_entries} entry e {num_envs} ambienti per ogni step.")
    del first_load
    gc.collect()

    steps_mmap = np.lib.format.open_memmap(f"{CACHE_DIR}/steps.npy", dtype='int32', mode='w+', shape=(num_entries,))
    quat_mmap = np.lib.format.open_memmap(f"{CACHE_DIR}/quat.npy", dtype='float32', mode='w+', shape=(num_entries, num_envs, 4))
    ang_diff_mmap = np.lib.format.open_memmap(f"{CACHE_DIR}/ang_diff.npy", dtype='float32', mode='w+', shape=(num_entries, num_envs))
    angvel_mmap = np.lib.format.open_memmap(f"{CACHE_DIR}/angvel.npy", dtype='float32', mode='w+', shape=(num_entries, num_envs, 3))
    angacc_mmap = np.lib.format.open_memmap(f"{CACHE_DIR}/angacc.npy", dtype='float32', mode='w+', shape=(num_entries, num_envs, 3))
    actions_mmap = np.lib.format.open_memmap(f"{CACHE_DIR}/actions.npy", dtype='float32', mode='w+', shape=(num_entries, num_envs, 3))

    checkpoint = torch.load(LOG_PATH, map_location="cpu", weights_only=True)
    
    for idx in range(num_entries):
        entry = checkpoint[idx]
        steps_mmap[idx] = entry["step"]
        quat_mmap[idx] = entry["quat"].numpy()
        ang_diff_mmap[idx] = entry["ang_diff"].numpy()
        angvel_mmap[idx] = entry["angvel"].numpy()
        angacc_mmap[idx] = entry["angacc"].numpy()
        actions_mmap[idx] = entry["actions"].numpy()
        
        checkpoint[idx] = None
        if idx % 500 == 0:
            steps_mmap.flush()
            quat_mmap.flush()
            ang_diff_mmap.flush()
            angvel_mmap.flush()
            angacc_mmap.flush()
            actions_mmap.flush()
            gc.collect()
            print(f"Progresso estrazione: {idx}/{num_entries}")

    del checkpoint
    gc.collect()
    print("Conversione completata con successo.")

build_cache_with_low_ram()

print("Lettura dati dalla cache...")
time = torch.from_numpy(np.load(f"{CACHE_DIR}/steps.npy", mmap_mode='r').copy() / HZ).numpy()
quat_all = torch.from_numpy(np.load(f"{CACHE_DIR}/quat.npy", mmap_mode='r'))
ang_diff_all = torch.from_numpy(np.load(f"{CACHE_DIR}/ang_diff.npy", mmap_mode='r')).unsqueeze(-1)
angvel_all = torch.from_numpy(np.load(f"{CACHE_DIR}/angvel.npy", mmap_mode='r'))
angacc_all = torch.from_numpy(np.load(f"{CACHE_DIR}/angacc.npy", mmap_mode='r'))
actions_all = torch.from_numpy(np.load(f"{CACHE_DIR}/actions.npy", mmap_mode='r'))

num_envs = quat_all.shape[1]
env_indices = random.sample(range(num_envs), min(N_ENV_PLOT, num_envs))

def smooth_time_axis(x, window):
    window = int(window) + (int(window) % 2 == 0)
    kernel = np.ones(window) / window
    pad = window // 2

    x_pad = np.pad(x, [(pad, pad)] + [(0, 0)] * (x.ndim - 1), mode="edge")

    return np.apply_along_axis(
        lambda y: np.convolve(y, kernel, mode="valid"),
        0,
        x_pad
    )

def plot_component(title, data_all, labels, non_negative=False, log_scale=False):
    print(f"Generazione plot per {title}...")

    if MAX_TIME is not None:
        cut_idx = np.searchsorted(time, MAX_TIME, side="right")
        data_all = data_all[:cut_idx, :, :]
        time_np = time[:cut_idx]
    else:
        time_np = time

    C = data_all.shape[2]
    N_steps = data_all.shape[0]
    num_sampled_envs = len(env_indices)

    cmap = plt.get_cmap("gist_rainbow")
    colors = [cmap(j / num_sampled_envs) for j in range(num_sampled_envs)]

    mean_np = np.zeros((N_steps, C), dtype=np.float32)
    std_np  = np.zeros((N_steps, C), dtype=np.float32)

    chunk_size = 5000
    for start in range(0, N_steps, chunk_size):
        end = min(start + chunk_size, N_steps)
        chunk = data_all[start:end, :, :].clone()
        mean_np[start:end] = chunk.mean(dim=1).numpy()
        std_np[start:end]  = chunk.std(dim=1).numpy()

    lower_np = mean_np - std_np
    upper_np = mean_np + std_np
    if non_negative:
        lower_np = lower_np.clip(min=0.0)

    sampled_data_np = data_all[:, env_indices, :].numpy()

    if SMOOTH_WINDOW is not None and SMOOTH_WINDOW > 1:
        sampled_data_np = smooth_time_axis(sampled_data_np, SMOOTH_WINDOW)
        
    scales = ["linear", "symlog"] if log_scale else ["linear"]

    for scale in scales:
        fig = plt.figure(figsize=(14, 3 * C))
        for i, label in enumerate(labels):
            ax = plt.subplot(C, 1, i + 1)

            segments = []
            plot_stride = max(1, N_steps // 2000)

            sampled_sliced = sampled_data_np[::plot_stride, :, i]
            time_sliced = time_np[::plot_stride]

            for j in range(num_sampled_envs):
                segments.append(np.column_stack((time_sliced, sampled_sliced[:, j])))
            
            line_segments = LineCollection(segments, colors=colors, alpha=0.8, linewidths=1.0, rasterized=True)
            ax.add_collection(line_segments)
            
            ax.set_xlim(min(time_np), max(time_np))
            ax.set_ylim(sampled_data_np[:, :, i].min(), sampled_data_np[:, :, i].max())

            plt.plot(time_np, mean_np[:, i], color='black', label='Mean', linewidth=1.5)
            plt.fill_between(time_np, lower_np[:, i], upper_np[:, i], color='grey', alpha=0.4)

            if scale == "symlog":
                plt.yscale("symlog", linthresh=1e0)
                plt.ylabel(label + " [log]")
            else:
                plt.ylabel(label)

            plt.grid(True, which='both', linestyle='--', alpha=0.5)
            if i == C - 1:
                plt.xlabel("Time [s]")

        plt.tight_layout()
        filename = f"_img/plots_trajectories_env/{title.replace(' ', '_').lower()}{'_log' if scale == 'symlog' else ''}.png"
        plt.savefig(filename, dpi=300)
        plt.close(fig)
        print(f"File salvato: {filename}")
        
    del mean_np, std_np, lower_np, upper_np, sampled_data_np
    gc.collect()

os.makedirs("_img/plots_trajectories_env", exist_ok=True)
plot_component("Quaternion", quat_all, ["x", "y", "z", "w"])
plot_component("Angular difference (deg)", ang_diff_all, ["angle (deg)"], non_negative=True, log_scale=True)
plot_component("Angular velocity", angvel_all, ["x", "y", "z"])
plot_component("Angular acceleration", angacc_all, ["x", "y", "z"])
plot_component("Actions", actions_all, ["x", "y", "z"], log_scale=True)
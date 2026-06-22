import argparse
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

HZ = 60

def nat_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))]

TAGS = ["Angular Error/q_diff_mean_deg", "Energy/mean",
        "Energy/delta_mean", "Goal/goal", "Torque/max_mean"]

LOG_DISPLAY = {
    "Angular Error/q_diff_mean_deg": 1e0,
    "Energy/mean": 1e1,
    "Energy/delta_mean": 1e0,
    "Torque/max_mean": 1e-1,
}

def normalize_eval_name(name):
    if name == "nominal": return "nominal"
    if name in ("random", "noise"): return "random-noise"
    return name

def load_data(files, method):
    step_data = defaultdict(lambda: defaultdict(list))
    for f in files:
        try:
            ea = EventAccumulator(str(f), size_guidance={"scalars": 0})
            ea.Reload()
            for t in set(ea.Tags().get("scalars", [])) & set(TAGS):
                for e in ea.Scalars(t):
                    step_data[t][e.step / HZ].append(e.value)
        except: continue
    
    result = {}
    for t in step_data:
        steps = sorted(step_data[t].keys())
        if method == "median":
            m = [np.median(step_data[t][s]) for s in steps]
        else:
            m = [np.mean(step_data[t][s]) for s in steps]
        mi = [np.min(step_data[t][s]) for s in steps]
        ma = [np.max(step_data[t][s]) for s in steps]
        result[t] = {"x": np.array(steps), "m": np.array(m), "min": np.array(mi), "max": np.array(ma)}
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--training", action="store_true")
    ap.add_argument("--method", choices=["median", "mean"], required=True)
    ap.add_argument("--outdir", default="_img/plots_reward_policy")
    args = ap.parse_args()

    base_path = Path(args.input)
    paths = list(base_path.rglob("*tfevents*"))
    
    grouped = defaultdict(list)
    for p in paths:
        rel = p.relative_to(base_path).parent

        if args.training:
            g = rel.parent
            gid = (g.parent / normalize_eval_name(g.name)).as_posix()
        else:
            eval_name = normalize_eval_name(rel.parent.parent.name)
            base_name = rel.parent.parent.parent.as_posix()
            gid = f"{base_name}/{eval_name}"

        if gid == ".":
            continue

        grouped[gid].append(p)
    
    print(f"grouped files: { {gid: len(files) for gid, files in grouped.items()} }")

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    
    #########################################################################
    
    all_results = {gid: load_data(files, args.method) for gid, files in grouped.items()}

    if args.training:
        unique_groups = sorted(all_results.keys(), key=nat_key)
    else:
        unique_groups = sorted(
            set(gid.rsplit('/', 1)[0] for gid in all_results.keys()),
            key=nat_key
        )
    
    cmap = plt.get_cmap("gist_rainbow", len(unique_groups))

    print(f"Found groups: {unique_groups}")
    for t in TAGS:
        plt.figure(figsize=(20, 8))
        
        # --- PATCH LOG SCALE ---
        if t in LOG_DISPLAY:
            plt.yscale("symlog", linthresh=LOG_DISPLAY[t])
        # -----------------------

        for i, gid in enumerate(sorted(all_results.keys(), key=nat_key)):
            if t not in all_results[gid]: continue

            if args.training:
                # TRAINING PLOTS: Each run gets a unique color
                color = cmap(i)
            else:
                # EVAL PLOTS: Group by training run, with nominal vs random-noise distinction
                train_name, eval_name = gid.rsplit('/', 1)
                group_idx = unique_groups.index(train_name)
                base_color = cmap(group_idx)
                alpha_val = 1.0 if eval_name == "nominal" else 0.5
                color = (base_color[0], base_color[1], base_color[2], alpha_val)

            d = all_results[gid][t]
            last_val = np.mean(d["m"][-max(1, int(len(d["m"]) * 0.05)):]) if len(d["m"]) > 0 else 0.0
            plt.plot(d["x"], d["m"], label=f"{gid} ({last_val:.2f})", color=color, linewidth=1.5)
            # plt.fill_between(d["x"], d["min"], d["max"], color=color, alpha=0.2)

        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend(fontsize='large', ncol=2, loc='upper right')
        plt.xlabel("Seconds")
        plt.ylabel("Value")
        plt.tight_layout()
        plt.savefig(out / f"{t.replace('/', '_')}_{'train' if args.training else 'eval'}.png", dpi=150)
        plt.close()

    # ================== PRINT TABLE ==================
    if args.training:
        base_runs = sorted(all_results.keys(), key=nat_key)
    else:
        base_runs = sorted(unique_groups, key=nat_key)

    sub_h = " | NOMINAL   | RANDOM-NOISE | DIFF%  "
    name_w = 40
    h_top = " " * name_w
    h_mid = f"{'RUN ID':<{name_w}}"
    
    for t in TAGS:
        h_top += f" | {t.split('/')[-1]:^34}"
        h_mid += sub_h

    print(f"\n{h_top}\n{h_mid}\n{'-' * len(h_mid)}")

    for base in base_runs:
        row = f"{base:<{name_w}}"

        if args.training:
            nominal_id = base
            noise_id = None
        else:
            nominal_id = f"{base}/nominal"
            noise_id = f"{base}/random-noise"

        for t in TAGS:
            d_nom = all_results.get(nominal_id, {}).get(t, {})
            d_noi = all_results.get(noise_id, {}).get(t, {}) if noise_id is not None else {}
            
            v_nom = np.mean(d_nom["m"][-max(1, int(len(d_nom["m"]) * 0.05)):]) if "m" in d_nom else None
            v_noi = np.mean(d_noi["m"][-max(1, int(len(d_noi["m"]) * 0.05)):]) if "m" in d_noi else None

            f = lambda v: f"{v:9.2e}" if v is not None and abs(v) > 9999 else (f"{v:9.2f}" if v is not None else "      N/A")

            s_nom = f(v_nom)
            s_noi = f(v_noi)

            s_diff = "        -"
            if v_nom is not None and v_noi is not None and v_nom != 0:
                diff = (v_noi - v_nom) / abs(v_nom) * 100
                s_diff = f"{diff:+8.1e}%" if abs(diff) > 999 else f"{diff:+8.1f}%"

            row += f" | {s_nom} | {s_noi} | {s_diff} "
        
        print(row)
    print("=" * len(h_mid))

if __name__ == "__main__":
    main()
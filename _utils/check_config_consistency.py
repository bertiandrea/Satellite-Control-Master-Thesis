import argparse
import copy
import json
import re
from collections import defaultdict
from pathlib import Path

# Regex per estrarre il timestamp di train dal nome del config
CFG_RE = re.compile(r"^config_(?P<eval_y>\d{8})_(?P<eval_tm>\d{6})_run_(?P<train_y>\d{8})_(?P<train_tm>\d{6})\.json$")

def get_train_ts(name: str) -> str:
    m = CFG_RE.search(name)
    if m:
        return f"{m.group('train_y')}_{m.group('train_tm')}"
    return "unknown"

def get_experiment_branch(path: Path, root_dir: Path) -> str:
    try:
        rel_path = path.resolve().relative_to(root_dir.resolve())
        return str(rel_path.parent)
    except ValueError:
        return ""

def normalize_config(d: dict) -> dict:
    data = copy.deepcopy(d)

    # 1. Ignora il seed
    if "seed" in data:
        data["seed"] = "IGNORED_SEED"

    if "rl" in data and "PPO" in data["rl"]:
        ppo = data["rl"]["PPO"]

        # 2. Ignora object_id (es. <function <lambda> at 0x7f98633791f0>)
        if "rewards_shaper" in ppo and "at 0x" in ppo["rewards_shaper"]:
                    ppo["rewards_shaper"] = "IGNORED_OBJECT_ID"
        
        # 3. Ignora timestamp negli esperimenti
        if "experiment" in ppo and "experiment_name" in ppo["experiment"]:
            ppo["experiment"]["experiment_name"] = "IGNORED_TIMESTAMP"

    # 4. Ignora timestamp nei percorsi di log
    if "log_status" in data and "log_dir" in data["log_status"]:
        data["log_status"]["log_dir"] = "IGNORED_TIMESTAMP"

    if "log_trajectories" in data and "log_file" in data["log_trajectories"]:
        data["log_trajectories"]["log_file"] = "IGNORED_TIMESTAMP"

    return data

def main():
    ap = argparse.ArgumentParser(description="Verifica che i config di un esperimento siano identici.")
    ap.add_argument("--root", default="~/Satellite-Control-Thesis-Baseline/eval/configs",
                    help="Directory radice dei configs (default: %(default)s)")
    args = ap.parse_args()

    cfg_root = Path(args.root).expanduser().resolve()
    
    if not cfg_root.is_dir():
        print(f"[ERRORE] Directory non trovata: {cfg_root}")
        return

    configs = [p for p in cfg_root.rglob("*.json") if p.is_file() and CFG_RE.search(p.name)]

    groups = defaultdict(list)
    for p in configs:
        branch = get_experiment_branch(p, cfg_root)
        train_ts = get_train_ts(p.name)
        groups[(branch, train_ts)].append(p)

    errors_found = False

    for (branch, train_ts), files in sorted(groups.items()):
        if len(files) < 2:
            continue
            
        print(f"Verifica gruppo: Run {train_ts} -> [{branch}] ({len(files)} file)")
        
        ref_file = sorted(files)[0]
        try:
            with open(ref_file, "r") as f:
                ref_data = normalize_config(json.load(f))
        except Exception as e:
            print(f"  [!] Errore di lettura config di riferimento {ref_file.name}: {e}")
            errors_found = True
            continue

        for test_file in sorted(files)[1:]:
            try:
                with open(test_file, "r") as f:
                    test_data = normalize_config(json.load(f))
            except Exception as e:
                print(f"  [!] Errore di lettura {test_file.name}: {e}")
                errors_found = True
                continue

            if ref_data != test_data:
                errors_found = True
                print(f"\n  [FAIL] Incoerenza trovata in: {test_file.name}")
                print(f"         (I dati non combaciano con la baseline: {ref_file.name})")
                    
    if not errors_found:
        print("\n[OK] Tutti i file di configurazione raggruppati sono perfettamente identici!")
    else:
        print("\n[FAIL] Trovate delle incongruenze (vedi log sopra).")

if __name__ == "__main__":
    main()
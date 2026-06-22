import argparse
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

# --- CONFIGURAZIONE ---
EXPECTED_SEEDS = [420, 4200, 42000, 420000, 4200000]
EXPECTED_COUNTS = Counter(EXPECTED_SEEDS)
EXPECTED_N = len(EXPECTED_SEEDS)

CFG_RE = re.compile(r"^config_(?P<eval_y>\d{8})_(?P<eval_tm>\d{6})_run_(?P<train_y>\d{8})_(?P<train_tm>\d{6})\.json$")
RUN_RE = re.compile(r"^run_(?P<y>\d{8})_(?P<tm>\d{6})$")
TRAJECTORIES_RE = re.compile(r"^trajectories_(?P<y>\d{8})_(?P<tm>\d{6})\.pt$")
STATUS_RE = re.compile(r"^status_(?P<y>\d{8})_(?P<tm>\d{6})$")

def dst_in_current_branch(src: Path, run_group_dir: str) -> Path:
    if src.parent.name == run_group_dir: return src
    return src.parent / run_group_dir / src.name

def get_eval_ts_cfg(name: str) -> str:
    m = CFG_RE.search(name)
    return f"{m.group('eval_y')}_{m.group('eval_tm')}"

def get_train_ts_cfg(name: str) -> str:
    m = CFG_RE.search(name)
    return f"{m.group('train_y')}_{m.group('train_tm')}"

def get_ts(name: str, regex: re.Pattern) -> str:
    m = regex.search(name)
    return f"{m.group('y')}_{m.group('tm')}"

def find_seed(data):
    if isinstance(data, dict):
        if isinstance(data.get("seed"), int):
            return data["seed"]
        for v in data.values():
            res = find_seed(v)
            if res is not None:
                return res
    elif isinstance(data, list):
        for v in data:
            res = find_seed(v)
            if res is not None:
                return res
    return None

def mv(src: Path, dst: Path, dry: bool) -> None:
    src = src.resolve()
    dst = dst.resolve()
    if src == dst:
        return
    if dry:
        print(f"[DRY] mv {src} -> {dst}")
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        print(f"[OK ] mv {src} -> {dst}")

def get_experiment_branch(path: Path, root_dir: Path) -> str:
    """Estrae la sottocartella relativa alla radice escludendo il nome del file stesso."""
    try:
        rel_path = path.resolve().relative_to(root_dir.resolve())
        # Ritorna il percorso della cartella madre come stringa (es. "randomization/0_05/nominal")
        return str(rel_path.parent)
    except ValueError:
        return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["audit", "move"], required=True,
                    help="audit: only checks, move: organize into run_<train_ts>")

    ap.add_argument("--root", default="~/Satellite-Control-Thesis-Baseline/eval",
                    help="Evaluating root (default: %(default)s)")
    ap.add_argument("--config-dir", default="configs",
                    help="relative to root (default: %(default)s)")
    ap.add_argument("--runs-dir", default="runs",
                    help="relative to root (default: %(default)s)")
    ap.add_argument("--trajectories-dir", default="trajectories",
                    help="relative to root (default: %(default)s)")
    ap.add_argument("--status-dir", default="status",
                    help="relative to root (default: %(default)s)")

    ap.add_argument("--dry-run", action="store_true",
                    help="only for move mode: print operations without moving")

    ap.add_argument("--no-trajectories", action="store_true",
                    help="exclude trajectories files from checks and moves")
    ap.add_argument("--no-runs", action="store_true",
                    help="exclude runs directories from checks and moves")
    ap.add_argument("--no-seed", action="store_true",
                    help="exclude seed checks")

    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()

    cfg_root = (root / args.config_dir).resolve()
    status_root = (root / args.status_dir).resolve()
    if not args.no_runs:
        runs_root = (root / args.runs_dir).resolve()
    if not args.no_trajectories:
        trajectories_root = (root / args.trajectories_dir).resolve()

    if not cfg_root.is_dir():
        raise SystemExit(f"config dir not found: {cfg_root}")
    if not status_root.is_dir():
        raise SystemExit(f"status dir not found: {status_root}")
    if not args.no_runs:
        if not runs_root.is_dir():
            raise SystemExit(f"runs dir not found: {runs_root}")
    if not args.no_trajectories:
        if not trajectories_root.is_dir():
            raise SystemExit(f"trajectories dir not found: {trajectories_root}")

    configs = [p for p in cfg_root.rglob("*.json") if p.is_file() and CFG_RE.search(p.name)]
    status_dirs = [d for d in status_root.rglob("*") if d.is_dir() and STATUS_RE.search(d.name)]
    if not args.no_runs:
        run_dirs = [d for d in runs_root.rglob("*") if d.is_dir() and RUN_RE.search(d.name)]
    if not args.no_trajectories:
        trajectories = [p for p in trajectories_root.rglob("*.pt") if p.is_file() and TRAJECTORIES_RE.search(p.name)]

    errors = []
    seeds_by_key = {}
    planned_moves = []

    cfg_by_eval_ts = {(get_experiment_branch(p, cfg_root), get_eval_ts_cfg(p.name)): p for p in configs}
    status_by_ts = {(get_experiment_branch(d, status_root), get_ts(d.name, STATUS_RE)): d for d in status_dirs}
    
    if not args.no_runs:
        runs_by_ts = {(get_experiment_branch(d, runs_root), get_ts(d.name, RUN_RE)): d for d in run_dirs}
    if not args.no_trajectories:
        trajectories_by_ts = {(get_experiment_branch(p, trajectories_root), get_ts(p.name, TRAJECTORIES_RE)): p for p in trajectories}

    for (branch, eval_ts), config in sorted(cfg_by_eval_ts.items()):
        train_ts = get_train_ts_cfg(config.name)
        
        group_key = f"{train_ts} [{branch}]"

        try:
            with open(config, "r") as f:
                data = json.load(f)
        except Exception as e:
            errors.append(f"Run {group_key}: Impossibile leggere {config.name} ({e})")
            continue

        if not args.no_seed:
            seed = find_seed(data)
            if seed is None:
                errors.append(f"Run {group_key}: Seed non trovato in {config.name}")
                continue
            seeds_by_key.setdefault(group_key, []).append(seed)

        status_dir = status_by_ts.get((branch, eval_ts))
        if status_dir is None:
            errors.append(f"Run {group_key}: Nessuna cartella STATUS trovata per {config.name} (atteso status_{eval_ts} in {branch})")
            continue
            
        if not args.no_runs:
            run_dir = runs_by_ts.get((branch, eval_ts))
            if run_dir is None:
                errors.append(f"Run {group_key}: Nessuna cartella RUN trovata per {config.name} (atteso run_{eval_ts} in {branch})")
                continue
                
        if not args.no_trajectories:
            trajectories_file = trajectories_by_ts.get((branch, eval_ts))
            if trajectories_file is None:
                errors.append(f"Run {group_key}: Nessun file TRAJECTORIES trovato per {config.name} (atteso trajectories_{eval_ts} in {branch})")
                continue

        if args.mode == "move":
            run_group_dir = f"run_{train_ts}"

            planned_moves.append((config, dst_in_current_branch(config, run_group_dir)))
            planned_moves.append((status_dir, dst_in_current_branch(status_dir, run_group_dir)))
            if not args.no_runs:
                planned_moves.append((run_dir, dst_in_current_branch(run_dir, run_group_dir)))
            if not args.no_trajectories:
                planned_moves.append((trajectories_file, dst_in_current_branch(trajectories_file, run_group_dir)))

    if not args.no_seed:
        for group_key, seeds in seeds_by_key.items():
            found_counts = Counter(seeds)

            if len(seeds) != EXPECTED_N:
                errors.append(
                    f"Run {group_key}: numero seed trovato {len(seeds)} (atteso {EXPECTED_N})"
                )

            if found_counts != EXPECTED_COUNTS:
                errors.append(
                    f"Run {group_key}: seed trovati {dict(found_counts)} (attesi {dict(EXPECTED_COUNTS)})"
                )

    if errors:
        print(f"\n[FAIL] {args.mode.upper()} fallito con {len(errors)} errori:")
        for e in errors:
            print(f" - {e}")
        sys.exit(1)

    if args.mode == "move":
        moved_src = set()
        for src, dst in planned_moves:
            src_res = src.resolve()
            if src_res in moved_src:
                continue
            mv(src, dst, args.dry_run)
            moved_src.add(src_res)

    print(f"\n[OK] {args.mode.upper()} completato")

if __name__ == "__main__":
    main()
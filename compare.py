"""End-to-end comparison harness for Layer Sharing vs. FLAKE.

Runs both scripts across a selectable set of CIFAR-10 architectures under
identical hyperparameters, then writes a single presentation-ready log.
The log is rewritten incrementally after every individual run, and a sidecar
state file lets the sweep resume from where it left off if it is interrupted.

Running ``python compare.py`` (no flags) will:

* For every selected model, launch ``layer_sharing.py`` and ``flake.py`` on
  the same CIFAR-10 Dirichlet split for ``--rounds`` rounds, save their JSON
  summaries + stdout logs, and update the human-readable log file.

* After each run completes, write the state file and the log so you can peek
  mid-sweep, and re-run later with the same arguments to continue.

Incremental log + resume:
    The harness maintains ``<log>.state.json`` next to the human log.  It
    is rewritten after every completed run.  If the harness crashes or you
    Ctrl-C out, re-run ``python compare.py`` with the SAME arguments and it
    will skip the runs that already completed and pick up where it left
    off.  To start fresh, pass ``--fresh`` (or delete the state file).

Long-running runs (MacBook lid closed):
    On macOS, closing the lid puts the machine to sleep and suspends this
    script.  To keep a long sweep running with the lid closed, launch it
    under ``caffeinate``::

        caffeinate -s python compare.py

    (Keep the laptop plugged in, too: sleep-on-battery is enforced by the
    kernel regardless of caffeinate.)

Usage::

    python compare.py                          # ls + flake sweep
    python compare.py --rounds 100             # longer run
    python compare.py --ls-models 1,2,3,4,5,6  # full layer_sharing model zoo
    python compare.py --fk-models 2            # flake SimpleCNN only
    python compare.py --fresh                  # ignore any existing state
    python compare.py --log presentation.log   # custom log path
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.resolve()

STATE_SCHEMA = 3


# Model catalogs (must mirror MODEL_NAME_MAP inside layer_sharing.py / flake.py)
LS_MODELS = {
    1: "SimpleCNN",
    2: "SimpleCNN10",
    3: "VGG11-BN",
    4: "VGG13-BN",
    5: "VGG16-BN",
    6: "ResNet-20-CIFAR",
}
FK_MODELS = {
    1: "PaperCNN",
    2: "SimpleCNN",
    3: "SimpleCNN10",
    4: "VGG11-BN",
    5: "VGG13-BN",
    6: "VGG16-BN",
    7: "ResNet-20-CIFAR",
}

# "Canonical" = the apples-to-apples SimpleCNN pair used for the head-to-head
# summary tables. Both map to SimpleCNN on their native side.
CANONICAL_LS_MODEL = 1
CANONICAL_FK_MODEL = 2

# layer_sharing_id -> flake_id for equivalent architectures (used by the
# per-architecture comparison table when --ls-models covers several).
LS_TO_FK = {1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7}


# Child-process launcher
def _run_child(label: str, cmd: list[str], env_extra: dict,
               results_path: Path, log_path: Path) -> tuple[int, dict | None, float]:
    """Run a child framework process, capturing its stdout/stderr to ``log_path``
    and parsing the JSON summary it drops at ``results_path``."""
    print(f"\n{'=' * 78}\n[{label}] launching: {' '.join(cmd)}\n{'=' * 78}")
    env = {**os.environ, **env_extra, "FL_RESULTS_JSON": str(results_path)}
    if results_path.exists():
        results_path.unlink()
    t0 = time.time()
    with log_path.open("w") as fh:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh,
                              stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    print(f"[{label}] exit={proc.returncode}, wall={elapsed:.1f}s, "
          f"log={log_path.name}")
    result = None
    if results_path.exists():
        try:
            result = json.loads(results_path.read_text())
        except Exception as e:
            print(f"[{label}] failed to parse {results_path.name}: {e}")
    else:
        print(f"[{label}] WARNING: no JSON results at {results_path.name}")
        try:
            tail = log_path.read_text().splitlines()[-30:]
            print(f"[{label}] last lines of {log_path.name}:")
            for line in tail:
                print(f"    {line}")
        except Exception:
            pass
    return proc.returncode, result, elapsed


def _common_env(args) -> dict:
    """Build the FL_* environment variables both frameworks read."""
    return {
        "FL_BATCH_SIZE": str(args.batch_size),
        "FL_EPOCHS_PER_ROUND": str(args.epochs),
        "FL_DIRICHLET_ALPHA": str(args.alpha),
        "FL_ROUNDS": str(args.rounds),
        "FL_DISABLE_EARLY_STOP": "1",
    }


# Per-round accuracy extraction from child stdout logs
_FLAKE_RE = re.compile(r"\[client (\d+)\] round (\d+):\s*acc=([\d.]+)%")
_LAYER_SHARING_RE = re.compile(
    r"Client (\d+)\s*-\s*Round (\d+):\s*Accuracy:\s*([\d.]+)%"
)


def _parse_round_accuracies(log_path: Path | None,
                            framework: str) -> list[tuple[int, float]]:
    """Return ``(round, mean_acc_across_clients)`` pairs in ascending order."""
    if log_path is None or not log_path.exists():
        return []
    text = log_path.read_text(errors="replace")
    per_round: dict[int, list[float]] = {}
    regex = _FLAKE_RE if framework == "flake" else _LAYER_SHARING_RE
    for match in regex.finditer(text):
        _, rnd, acc = match.groups()
        per_round.setdefault(int(rnd), []).append(float(acc))
    return [(r, sum(vs) / len(vs)) for r, vs in sorted(per_round.items())]


def _rounds_to_target(history: list[tuple[int, float]], target: float):
    """First round at which the mean accuracy reaches ``target``%. Else ``None``."""
    for r, a in history:
        if a >= target:
            return r
    return None


# Row normalisation
def _row_for(result: dict | None, framework: str) -> dict:
    """Pull the comparable metrics out of a framework's JSON summary."""
    if result is None:
        return {"final_acc": None, "best_acc": None, "wall_s": None,
                "best_round": None, "model": "-",
                "train_s": None, "comm_s": None,
                "metric": "-"}
    if framework == "layer_sharing":
        pc = result.get("per_client") or {}
        train_vals = [v.get("training_s") for v in pc.values()
                      if isinstance(v.get("training_s"), (int, float))]
        comm_vals = [(v.get("comm_io_s") or 0.0) + (v.get("comm_phase_s") or 0.0)
                     for v in pc.values()]
        best_rounds = [v.get("best_round") for v in pc.values()
                       if isinstance(v.get("best_round"), (int, float))
                       and v.get("best_round") >= 0]
        return {
            "final_acc": result.get("avg_final_acc"),
            "best_acc": result.get("avg_best_acc"),
            "wall_s": result.get("total_time_s"),
            "best_round": (sum(best_rounds) / len(best_rounds)) if best_rounds else None,
            "model": result.get("model", "-"),
            "train_s": (sum(train_vals) / len(train_vals)) if train_vals else None,
            "comm_s": (sum(comm_vals) / len(comm_vals)) if comm_vals else None,
            "metric": "avg over local clients",
        }
    if framework == "flake":
        pc = result.get("per_client") or {}
        train_vals = [v.get("training_s") for v in pc.values()
                      if isinstance(v.get("training_s"), (int, float))]
        comm_vals = [v.get("comm_s") for v in pc.values()
                     if isinstance(v.get("comm_s"), (int, float))]
        best_rounds = [v.get("best_round") for v in pc.values()
                       if isinstance(v.get("best_round"), (int, float))
                       and v.get("best_round") >= 0]
        return {
            "final_acc": result.get("avg_final_acc"),
            "best_acc": result.get("avg_best_acc"),
            "wall_s": result.get("total_time_s"),
            "best_round": (sum(best_rounds) / len(best_rounds)) if best_rounds else None,
            "model": result.get("model", "-"),
            "train_s": (sum(train_vals) / len(train_vals)) if train_vals else None,
            "comm_s": (sum(comm_vals) / len(comm_vals)) if comm_vals else None,
            "metric": "avg over local clients",
        }
    return {"final_acc": None, "best_acc": None, "wall_s": None,
            "best_round": None, "model": "-",
            "train_s": None, "comm_s": None, "metric": "-"}


def _fmt_acc(v) -> str:
    return f"{v:6.2f}%" if isinstance(v, (int, float)) else "   -  "


def _fmt_secs(v) -> str:
    return f"{v:7.1f}s" if isinstance(v, (int, float)) else "    -  "


def _fmt_round(v) -> str:
    if isinstance(v, (int, float)):
        return f"r={int(round(v)):<3d}"
    return "  -  "


# State file (incremental save / resume)
def _state_path_for(log_path: Path) -> Path:
    """Return the sidecar state file path for a given human-log path."""
    return log_path.with_suffix(log_path.suffix + ".state.json")


def _args_fingerprint(args) -> dict:
    """Arguments that must match for a resume to be valid (paths excluded)."""
    return {
        "rounds": args.rounds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "alpha": args.alpha,
        "ls_models": list(args.ls_models),
        "fk_models": list(args.fk_models),
        "skip_layer_sharing": bool(args.skip_layer_sharing),
        "skip_flake": bool(args.skip_flake),
    }


def _entry_to_state(entry: dict) -> dict:
    """Serialise one completed-run entry for the state file."""
    return {
        "result":   entry.get("result"),
        "row":      entry.get("row"),
        "history":  [[int(r), float(a)] for r, a in entry.get("history", [])],
        "log_path": str(entry.get("log_path")) if entry.get("log_path") else None,
        "json_path": str(entry.get("json_path")) if entry.get("json_path") else None,
        "elapsed":  entry.get("elapsed"),
    }


def _entry_from_state(raw: dict | None) -> dict | None:
    """Inverse of :func:`_entry_to_state`. Paths become :class:`Path` objects."""
    if raw is None:
        return None
    return {
        "result":   raw.get("result"),
        "row":      raw.get("row"),
        "history":  [(int(r), float(a)) for r, a in raw.get("history", [])],
        "log_path": Path(raw["log_path"]) if raw.get("log_path") else None,
        "json_path": Path(raw["json_path"]) if raw.get("json_path") else None,
        "elapsed":  raw.get("elapsed"),
    }


def _load_state(state_path: Path):
    """Read the sweep's saved state. Returns ``(sweep, fingerprint, targets)``
    or ``(None, None, None)`` if the file is missing or an older schema."""
    if not state_path.exists():
        return None, None, None
    try:
        data = json.loads(state_path.read_text())
    except Exception as e:
        print(f"[resume] failed to parse {state_path.name}: {e}; ignoring.")
        return None, None, None
    if data.get("schema") != STATE_SCHEMA:
        print(f"[resume] {state_path.name} uses schema "
              f"{data.get('schema')!r}, expected {STATE_SCHEMA}; ignoring.")
        return None, None, None

    sweep = {"layer_sharing": {}, "flake": {}}
    for fw in sweep:
        for mid_str, raw in (data.get("clean", {}).get(fw, {}) or {}).items():
            entry = _entry_from_state(raw)
            if entry:
                sweep[fw][int(mid_str)] = entry

    return sweep, data.get("args"), data.get("targets")


def _save_state(state_path: Path, args, sweep: dict,
                targets: list[float], completed: bool) -> None:
    """Atomically write the sweep's state (fingerprint + every completed run)."""
    data = {
        "schema": STATE_SCHEMA,
        "args": _args_fingerprint(args),
        "targets": list(targets) if targets else [],
        "last_updated": datetime.now().isoformat(timespec="seconds"),
        "completed": bool(completed),
        "clean": {
            fw: {str(mid): _entry_to_state(entry)
                 for mid, entry in runs.items()}
            for fw, runs in sweep.items()
        },
    }
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.replace(state_path)


# Log-file writer
def _write_presentation_log(path: Path, args, sweep: dict,
                            targets: list[float]) -> None:
    """Rewrite the human-readable slide log from the in-memory sweep state.

    ``sweep[framework][model_id] = {result, row, history, log_path, json_path, elapsed}``.
    """

    with path.open("w") as fh:
        w = fh.write

        # --- Header ---
        w("=" * 78 + "\n")
        w("FLAKE vs Layer Sharing -- experimental results\n")
        w("(sections are labelled with the presentation slide they feed into)\n")
        w("=" * 78 + "\n")
        w(f"Generated              : {datetime.now().isoformat(timespec='seconds')}\n")
        w(f"Host / python          : {platform.platform()} / {platform.python_version()}\n")
        w(f"Log file               : {path.name}\n")
        w(f"Rounds per run         : {args.rounds}\n")
        clean_done = sum(len(v) for v in sweep.values())
        clean_planned = ((0 if args.skip_layer_sharing else len(args.ls_models))
                         + (0 if args.skip_flake else len(args.fk_models)))
        w(f"Runs executed          : {clean_done}/{clean_planned}\n")
        w("\n")

        # --- Slide 14: experimental setup ---
        sample = None
        for fw, runs in sweep.items():
            for mid, entry in runs.items():
                if entry.get("result"):
                    sample = entry["result"]
                    break
            if sample:
                break
        if sample is None:
            sample = {}

        w("-" * 78 + "\n")
        w("[SLIDE 14 -- Experimental Setup]\n")
        w("-" * 78 + "\n")
        w(f"  dataset               : CIFAR-10\n")
        w(f"  partition             : class-aware Dirichlet\n")
        w(f"  dirichlet_alpha       : {sample.get('dirichlet_alpha', args.alpha)}\n")
        w(f"  num_clients (N)       : {sample.get('n_clients', '-')}\n")
        w(f"  num_machines (M)      : {sample.get('n_machines', 1)}\n")
        w(f"  rounds (T)            : {args.rounds}\n")
        w(f"  batch_size            : {sample.get('batch_size', args.batch_size)}\n")
        w(f"  epochs_per_round      : "
          f"{sample.get('epochs_per_round', args.epochs)}\n")
        w(f"  optimizer             : SGD + momentum 0.9, lr 0.01\n")
        w(f"  KD temperature (tau)  : 1.0\n")
        w(f"  KD weight     (mu)    : 1.5 (ramped 0->1.5 over first 3 rounds in flake)\n")
        w(f"  layer_sharing models  : "
          f"{', '.join(LS_MODELS[m] for m in args.ls_models) or '(none)'}\n")
        w(f"  flake models          : "
          f"{', '.join(FK_MODELS[m] for m in args.fk_models) or '(none)'}\n")
        w("\n")

        # --- Slide 15: canonical head-to-head ---
        w("-" * 78 + "\n")
        w("[SLIDE 15 -- Final Accuracy (canonical SimpleCNN)]\n")
        w("-" * 78 + "\n")
        canonical_rows = [
            ("layer_sharing", sweep["layer_sharing"].get(CANONICAL_LS_MODEL)),
            ("flake",         sweep["flake"].get(CANONICAL_FK_MODEL)),
        ]
        w(f"  {'framework':<14} {'final':>8} {'best':>8} {'wall':>9}   metric\n")
        w(f"  {'-' * 14} {'-' * 8} {'-' * 8} {'-' * 9}   {'-' * 25}\n")
        valid = []
        for fw, entry in canonical_rows:
            row = entry["row"] if entry else _row_for(None, fw)
            w(f"  {fw:<14} "
              f"{_fmt_acc(row['final_acc'])} {_fmt_acc(row['best_acc'])} "
              f"{_fmt_secs(row['wall_s'])}   {row.get('metric', '-')}\n")
            if isinstance(row["final_acc"], (int, float)):
                valid.append((fw, row))
        if len(valid) == 2:
            winner_name, winner_row = max(valid, key=lambda kv: kv[1]["final_acc"])
            other = [v for v in valid if v[0] != winner_name][0]
            gap = winner_row["final_acc"] - other[1]["final_acc"]
            w(f"\n  winner (final acc)    : {winner_name} @ "
              f"{winner_row['final_acc']:.2f}%\n")
            w(f"  gap over {other[0]:<13} : +{gap:.2f} pp\n")
        w("\n  Note: both frameworks report the AVERAGE test accuracy across\n")
        w("        local clients on the same CIFAR-10 test set with the same seed.\n\n")

        # --- Slide 16: convergence (canonical) ---
        w("-" * 78 + "\n")
        w("[SLIDE 16 -- Convergence: Rounds to Target Accuracy (canonical SimpleCNN)]\n")
        w("-" * 78 + "\n")
        hist = {
            "layer_sharing": (sweep["layer_sharing"].get(CANONICAL_LS_MODEL) or {}).get("history", []),
            "flake":         (sweep["flake"].get(CANONICAL_FK_MODEL) or {}).get("history", []),
        }
        w(f"  target     layer_sharing   flake\n")
        w(f"  ------     -------------   -----\n")
        for tgt in targets:
            parts = [f"  >={tgt:4.0f}%    "]
            for name in ("layer_sharing", "flake"):
                r = _rounds_to_target(hist[name], tgt)
                parts.append(f"{_fmt_round(r):<15s} ")
            w("".join(parts).rstrip() + "\n")

        w("\n  Per-round accuracy trace (canonical SimpleCNN):\n")
        w(f"  {'round':>5}  {'layer_sharing':>13}  {'flake':>9}\n")
        all_rounds = set()
        for seq in hist.values():
            for r, _ in seq:
                all_rounds.add(r)
        round_map = {n: dict(seq) for n, seq in hist.items()}
        for r in sorted(all_rounds):
            ls_v = round_map["layer_sharing"].get(r)
            fk_v = round_map["flake"].get(r)
            def _f(v):
                return f"{v:9.2f}" if isinstance(v, (int, float)) else "      -  "
            w(f"  {r:>5}  {_f(ls_v):>13}  {_f(fk_v):>9}\n")
        w("\n")

        # --- Per-architecture head-to-head ---
        pairs = []
        for ls_id, fk_id in LS_TO_FK.items():
            ls_entry = sweep["layer_sharing"].get(ls_id)
            fk_entry = sweep["flake"].get(fk_id)
            if ls_entry or fk_entry:
                pairs.append((ls_id, fk_id, ls_entry, fk_entry))
        if pairs:
            w("-" * 78 + "\n")
            w("[SLIDE 15b -- Per-architecture head-to-head]\n")
            w("-" * 78 + "\n")
            w(f"  {'architecture':<18} "
              f"{'LS final':>9} {'FK final':>9} {'delta':>9}   "
              f"{'LS wall':>9} {'FK wall':>9}\n")
            w(f"  {'-' * 18} {'-' * 9} {'-' * 9} {'-' * 9}   "
              f"{'-' * 9} {'-' * 9}\n")
            gains = []
            for ls_id, fk_id, le, fe in pairs:
                arch = LS_MODELS[ls_id]
                l_row = le["row"] if le else _row_for(None, "layer_sharing")
                f_row = fe["row"] if fe else _row_for(None, "flake")
                la, fa = l_row.get("final_acc"), f_row.get("final_acc")
                if isinstance(la, (int, float)) and isinstance(fa, (int, float)):
                    delta = f"{fa - la:+6.2f}pp"
                    gains.append(fa - la)
                else:
                    delta = "    -  "
                w(f"  {arch:<18} "
                  f"{_fmt_acc(la):>9} {_fmt_acc(fa):>9} {delta:>9}   "
                  f"{_fmt_secs(l_row['wall_s']):>9} {_fmt_secs(f_row['wall_s']):>9}\n")
            if gains:
                mean_gain = sum(gains) / len(gains)
                wins = sum(1 for g in gains if g > 0)
                w(f"\n  FLAKE wins {wins}/{len(gains)} architectures; "
                  f"mean gain = {mean_gain:+.2f}pp\n")
            w("\n")

        # --- Per-system sweep tables ---
        _write_sweep_section(w, "layer_sharing", LS_MODELS,
                             sweep["layer_sharing"], targets, has_train_comm=True)
        _write_sweep_section(w, "flake", FK_MODELS,
                             sweep["flake"], targets, has_train_comm=True)

        # --- Cross-system headline (canonical) ---
        w("-" * 78 + "\n")
        w("[SLIDES 13 / 18 / 20 -- Headline Numbers (canonical SimpleCNN)]\n")
        w("-" * 78 + "\n")
        ls_fin = (canonical_rows[0][1] or {}).get("row", {}).get("final_acc") \
            if canonical_rows[0][1] else None
        fk_fin = (canonical_rows[1][1] or {}).get("row", {}).get("final_acc") \
            if canonical_rows[1][1] else None
        w("  - Both systems are peer-to-peer and server-free.\n")
        w("  - layer_sharing uses random layer stacking from peers, no KD.\n")
        w("  - flake adds KD against a locally-aggregated peer teacher on\n")
        w("    top of the same decentralised transport.\n\n")
        if all(isinstance(v, (int, float)) for v in (ls_fin, fk_fin)):
            w(f"  layer_sharing  final acc = {ls_fin:.2f}%  (baseline P2P)\n")
            w(f"  flake          final acc = {fk_fin:.2f}%  (P2P + KD, gap vs "
              f"layer_sharing = {fk_fin - ls_fin:+.2f}pp)\n")
        w("\n")

        # --- Appendix: file listing ---
        w("-" * 78 + "\n")
        w("[Appendix -- per-run artefacts]\n")
        w("-" * 78 + "\n")
        for fw, runs in sweep.items():
            for mid, entry in sorted(runs.items()):
                jp = entry.get("json_path")
                lp = entry.get("log_path")
                mn = entry.get("row", {}).get("model", "-")
                w(f"  {fw:<14} m={mid} ({mn:<16}): "
                  f"json={jp.name if jp else '-':<48} "
                  f"log={lp.name if lp else '-'}\n")
        w("\n")

        # --- Resume pointer ---
        state_path = _state_path_for(path)
        w("-" * 78 + "\n")
        w("[Resume state]\n")
        w("-" * 78 + "\n")
        w(f"  state file  : {state_path.name}\n")
        w(f"  To resume   : python compare.py   (same args)\n")
        w(f"  To restart  : python compare.py --fresh   (or rm the state file)\n")
        w("\n")


def _write_sweep_section(w, framework: str, model_map: dict,
                         runs: dict, targets: list[float],
                         has_train_comm: bool) -> None:
    """Write one per-framework table covering every model in the sweep."""
    title_map = {
        "layer_sharing": "Layer Sharing (decentralised P2P, random layer stacking, no KD)",
        "flake":         "FLAKE (decentralised P2P, KD vs locally-aggregated peer)",
    }
    w("-" * 78 + "\n")
    w(f"[SWEEP -- {framework}: {len(runs)} runs]\n")
    w(f"  {title_map.get(framework, framework)}\n")
    w("-" * 78 + "\n")
    if not runs:
        w("  (no runs in this sweep -- all skipped)\n\n")
        return

    col_targets = targets[-4:] if len(targets) > 4 else targets

    hdr = f"  {'model':<18} {'final':>8} {'best':>8} {'best_r':>7} {'wall':>9}"
    if has_train_comm:
        hdr += f" {'train':>9} {'comm':>9}"
    for t in col_targets:
        hdr += f" {'>=' + f'{int(t)}%':>6}"
    w(hdr + "\n")
    w("  " + "-" * (len(hdr) - 2) + "\n")

    for mid in sorted(runs.keys()):
        entry = runs[mid]
        row = entry["row"]
        hist = entry.get("history", [])
        line = (
            f"  {model_map.get(mid, f'm={mid}'):<18} "
            f"{_fmt_acc(row['final_acc'])} {_fmt_acc(row['best_acc'])} "
            f"{_fmt_round(row['best_round']):>7s} "
            f"{_fmt_secs(row['wall_s'])}"
        )
        if has_train_comm:
            line += f" {_fmt_secs(row['train_s'])} {_fmt_secs(row['comm_s'])}"
        for t in col_targets:
            r = _rounds_to_target(hist, t)
            line += f" {_fmt_round(r):>6s}"
        w(line + "\n")
    w("\n")


# Scenario runner
def _run_framework_model(framework: str, model_id: int,
                         args, common_env: dict, input_path: Path) -> dict:
    """Execute a single ``(framework, model_id)`` run and return a dict with
    the parsed JSON summary, normalised row, per-round history, paths and
    elapsed wall time."""
    stem = f"compare_{framework}_m{model_id}"
    log_path = ROOT / f"{stem}.log"
    json_path = ROOT / f"{stem}_results.json"

    if framework == "layer_sharing":
        cmd = [sys.executable, "layer_sharing.py", "--model", str(model_id)]
        env = {**common_env, "LAYER_SHARING_INPUT": str(input_path)}
    elif framework == "flake":
        cmd = [sys.executable, "flake.py", "--model", str(model_id)]
        env = {**common_env, "FLAKE_INPUT": str(input_path)}
    else:
        raise ValueError(framework)

    label = f"{framework} m={model_id}"
    _, result, elapsed = _run_child(label, cmd, env, json_path, log_path)
    history = _parse_round_accuracies(log_path, framework)
    return {
        "result":   result,
        "row":      _row_for(result, framework),
        "history":  history,
        "log_path": log_path,
        "json_path": json_path,
        "elapsed":  elapsed,
    }


def _entry_is_usable(entry: dict | None) -> bool:
    """Did a previously-saved entry actually succeed? If not, we re-run it."""
    if not entry:
        return False
    row = entry.get("row") or {}
    return isinstance(row.get("final_acc"), (int, float))


def _run_sweep(args, inputs, common_env, sweep: dict, checkpoint) -> dict:
    """Sweep every requested model for every non-skipped framework.

    ``sweep`` is the in-memory accumulator (pre-populated from the saved
    state file). ``checkpoint`` is called after each completed run so state
    + log stay up to date.
    """
    plan = []
    if not args.skip_layer_sharing:
        for mid in args.ls_models:
            plan.append(("layer_sharing", mid))
    if not args.skip_flake:
        for mid in args.fk_models:
            plan.append(("flake", mid))

    todo = [(fw, mid) for fw, mid in plan
            if not _entry_is_usable(sweep.get(fw, {}).get(mid))]
    already = len(plan) - len(todo)

    print(f"\nSweep: {len(plan)} runs planned, "
          f"{already} already completed from previous state, "
          f"{len(todo)} to run ({args.rounds} rounds each)")
    if already:
        for fw, mid in plan:
            if _entry_is_usable(sweep.get(fw, {}).get(mid)):
                mn = (LS_MODELS if fw == "layer_sharing" else FK_MODELS).get(
                    mid, f"m{mid}")
                row = sweep[fw][mid]["row"]
                fa = row.get("final_acc")
                fa_s = f"{fa:.2f}%" if isinstance(fa, (int, float)) else "-"
                print(f"    [skip] {fw} / {mn}  (resumed, final={fa_s})")

    for idx, (fw, mid) in enumerate(todo, start=1):
        mn = (LS_MODELS if fw == "layer_sharing" else FK_MODELS).get(
            mid, f"m{mid}")
        print(f"  [{idx}/{len(todo)}] {fw} / {mn}")
        entry = _run_framework_model(fw, mid, args, common_env, inputs[fw])
        sweep[fw][mid] = entry
        checkpoint()
    return sweep


# Arg parsing helpers
def _parse_model_list(spec: str, allowed: set[int]) -> list[int]:
    """Parse a comma-separated model-id list, enforcing the ``allowed`` set."""
    if not spec:
        return []
    out = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        mid = int(chunk)
        if mid not in allowed:
            raise argparse.ArgumentTypeError(
                f"model id {mid} not in {sorted(allowed)}"
            )
        out.append(mid)
    return out


def _print_lid_closed_hint() -> None:
    """Remind macOS users to wrap the harness in ``caffeinate`` for long runs."""
    if platform.system() != "Darwin":
        return
    has_caffeinate = shutil.which("caffeinate") is not None
    print("\n" + "-" * 78)
    print("NOTE -- keeping this sweep running with the MacBook lid CLOSED:")
    print("  Closing the lid puts macOS to sleep by default, which will")
    print("  suspend this run. To prevent that, relaunch under caffeinate:")
    if has_caffeinate:
        print("      caffeinate -s python compare.py ...")
    else:
        print("      caffeinate -s python compare.py ...   "
              "(`caffeinate` is built into macOS)")
    print("  Also keep the laptop plugged in -- sleep-on-battery is enforced")
    print("  by the kernel and caffeinate cannot override it.")
    print("  Safe to Ctrl-C: re-run `python compare.py` to resume where it")
    print("  left off (state file: <log>.state.json).")
    print("-" * 78 + "\n")


# Entry point
def main() -> int:
    """Parse args, load / init the sweep state, run any missing combos, and
    write the human-readable log + state file after every run."""
    parser = argparse.ArgumentParser(
        description="Run layer_sharing.py and flake.py across every selected "
                    "model, writing an incremental, resumable "
                    "presentation-ready log.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "layer_sharing models: "
            + ", ".join(f"{k}={v}" for k, v in LS_MODELS.items())
            + "\nflake models       : "
            + ", ".join(f"{k}={v}" for k, v in FK_MODELS.items())
            + "\n\nResume: re-run with the same args. Pass --fresh to wipe state."
              "\n\nTo keep the run alive with the MacBook lid closed:"
              "\n    caffeinate -s python compare.py ..."
        ),
    )
    parser.add_argument("--rounds", type=int, default=40,
                        help="FL rounds per run (default: 40)")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Local epochs per round (default: 1)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Local SGD batch size (default: 32)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Dirichlet alpha for non-IID split (default: 0.5)")

    # Defaults cover one small CNN, one medium CNN, and ResNet-20 on both
    # systems so each layer_sharing run has a direct flake counterpart with
    # the same architecture. VGG variants are expensive (10-15M params) so
    # they're opt-in.
    parser.add_argument(
        "--ls-models", type=str,
        default="1,2,6",
        help="Comma-separated layer_sharing --model IDs to sweep "
             "(default: 1,2,6 = SimpleCNN, SimpleCNN10, ResNet-20, chosen "
             "to mirror the flake defaults). Full zoo: '1,2,3,4,5,6' adds "
             "VGG11/13/16-BN.",
    )
    parser.add_argument(
        "--fk-models", type=str,
        default="2,3,7",
        help="Comma-separated flake --model IDs to sweep "
             "(default: 2,3,7 = SimpleCNN, SimpleCNN10, ResNet-20, chosen "
             "to mirror the layer_sharing defaults). Full zoo: "
             "'1,2,3,4,5,6,7' adds PaperCNN + VGG11/13/16-BN. VGG variants "
             "are ~10-15M params and take hours per run.",
    )

    parser.add_argument("--skip-layer-sharing", action="store_true")
    parser.add_argument("--skip-flake", action="store_true")
    parser.add_argument("--log", type=str, default="presentation_results.log",
                        help="Combined slide-ready log path "
                             "(default: presentation_results.log).")
    parser.add_argument(
        "--targets", type=str, default="",
        help="Comma-separated accuracy targets (percent) for rounds-to-target "
             "columns. Default: 20, 30, 40, 50, 60.",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore any existing state file and start the sweep over. "
             "Per-run stdout logs (compare_*_m*.log) will also be overwritten.",
    )
    args = parser.parse_args()

    try:
        args.ls_models = _parse_model_list(args.ls_models, set(LS_MODELS.keys()))
        args.fk_models = _parse_model_list(args.fk_models, set(FK_MODELS.keys()))
    except argparse.ArgumentTypeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    inputs = {
        "layer_sharing": ROOT / "layer_sharing_input_localhost.txt",
        "flake":         ROOT / "flake_input_localhost.txt",
    }
    skip = {
        "layer_sharing": args.skip_layer_sharing,
        "flake":         args.skip_flake,
    }
    for name, path in inputs.items():
        if not skip[name] and not path.exists():
            print(f"Missing {path.name}; aborting.")
            return 1

    _print_lid_closed_hint()

    log_path = (ROOT / args.log) if not os.path.isabs(args.log) else Path(args.log)
    state_path = _state_path_for(log_path)

    # --- Resume scaffolding ---
    sweep: dict = {"layer_sharing": {}, "flake": {}}
    saved_targets: list[float] = []

    if args.fresh and state_path.exists():
        print(f"[fresh] deleting existing state {state_path.name}")
        state_path.unlink()

    if state_path.exists():
        loaded_sweep, saved_fp, saved_targets_raw = _load_state(state_path)
        if loaded_sweep is not None:
            current_fp = _args_fingerprint(args)
            if saved_fp != current_fp:
                diffs = [k for k in set(saved_fp) | set(current_fp)
                         if saved_fp.get(k) != current_fp.get(k)]
                print("[resume] ERROR: state file's arguments differ from the "
                      "current invocation:")
                for k in sorted(diffs):
                    print(f"         {k}: saved={saved_fp.get(k)!r}  "
                          f"current={current_fp.get(k)!r}")
                print("         Re-run with matching arguments, or pass "
                      "--fresh to start over.")
                return 2
            sweep = loaded_sweep
            saved_targets = saved_targets_raw or []
            clean_n = sum(len(v) for v in sweep.values())
            print(f"[resume] loaded state from {state_path.name}: "
                  f"{clean_n} completed run(s)")

    common_env = _common_env(args)
    print("Aligned settings for this comparison run:")
    for k, v in common_env.items():
        print(f"  {k} = {v}")
    print(f"  layer_sharing models = {args.ls_models}  "
          f"({', '.join(LS_MODELS[m] for m in args.ls_models)})")
    print(f"  flake         models = {args.fk_models}  "
          f"({', '.join(FK_MODELS[m] for m in args.fk_models)})")

    # --- Compute targets (for log columns). Recompute as new runs arrive. ---
    def _compute_targets() -> list[float]:
        if args.targets:
            try:
                return sorted({float(x) for x in args.targets.split(",")
                               if x.strip()})
            except ValueError:
                print(f"Ignoring malformed --targets {args.targets!r}")
        finals = []
        for fw in ("layer_sharing", "flake"):
            for entry in sweep.get(fw, {}).values():
                v = entry.get("row", {}).get("final_acc")
                if isinstance(v, (int, float)):
                    finals.append(v)
        top = max(finals) if finals else 60.0
        top5 = max(20.0, 5.0 * math.floor(top / 5.0))
        step = max(5.0, (top5 - 20.0) / 4.0)
        tgts = [round(20.0 + i * step, 1) for i in range(5)]
        if top5 not in tgts:
            tgts.append(top5)
        return sorted(set(tgts))

    # --- Incremental save + log rewrite ---
    def _checkpoint() -> None:
        tgts = _compute_targets()
        _save_state(state_path, args, sweep, tgts, completed=False)
        _write_presentation_log(log_path, args, sweep, tgts)

    # Write an initial log right away so the user can see planned structure.
    _checkpoint()

    t_harness_0 = time.time()

    # --- Run (or resume) the clean sweep ---
    _run_sweep(args, inputs, common_env, sweep, _checkpoint)

    # --- Final log (mark completed in state) ---
    final_targets = _compute_targets()
    _save_state(state_path, args, sweep, final_targets, completed=True)
    _write_presentation_log(log_path, args, sweep, final_targets)

    total = time.time() - t_harness_0
    print(f"\n>>> harness wall time = {total:.1f}s ({total/60:.1f} min)")
    print(f">>> wrote presentation-ready log to {log_path}")
    print(f">>> state file (safe to delete to reset): {state_path}")
    print(f">>> per-run stdout logs + JSONs: compare_*_m*.log / compare_*_m*_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

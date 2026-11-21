"""Summarise IGCA runs into the README results table.

Reads every runs/*/run_manifest.json and prints, per model and variant, the
mean and spread over seeds of: step-0 shift (nats), best held-out loss,
held-out improvement over the frozen backbone, and mean gate at the best step.

Usage (from the repository root):
    python summarize_runs.py            # all runs under runs/
    python summarize_runs.py runs_7b    # a different folder
"""
import glob
import json
import os
import statistics as st
import sys


def fmt(values, digits=4):
    vals = [v for v in values if v is not None]
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:.{digits}f}"
    return f"{st.mean(vals):.{digits}f} ± {st.stdev(vals):.{digits}f}"


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "runs"
    groups = {}
    for path in sorted(glob.glob(os.path.join(root, "*", "run_manifest.json"))):
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        if "smoke" in path:
            continue
        key = (m.get("model_id"), m.get("variant"))
        groups.setdefault(key, []).append(m)

    if not groups:
        print(f"no run_manifest.json found under {root}/")
        return 1

    print("| Model | Variant | Seeds | Step-0 shift (nats) | Best held-out loss | "
          "Gain vs frozen (nats) | Gain (%) | Mean gate at best |")
    print("|---|---|:-:|:-:|:-:|:-:|:-:|:-:|")
    for (model, variant), runs in sorted(groups.items()):
        seeds = ",".join(str(r.get("seed")) for r in sorted(runs, key=lambda r: r.get("seed", 0)))
        print(f"| {model} | {variant} | {seeds} "
              f"| {fmt([r.get('step0_shift_nats') for r in runs])} "
              f"| {fmt([r.get('best_heldout_gate_on') for r in runs])} "
              f"| {fmt([r.get('heldout_delta_nats') for r in runs])} "
              f"| {fmt([r.get('heldout_delta_pct') for r in runs], 1)} "
              f"| {fmt([r.get('mean_gate_at_best') for r in runs])} |")

    print("\nPer run:")
    for (model, variant), runs in sorted(groups.items()):
        for r in sorted(runs, key=lambda r: r.get("seed", 0)):
            print(f"  {model} {variant} seed={r.get('seed')}: frozen {r.get('heldout_gate_off'):.4f} "
                  f"-> best {r.get('best_heldout_gate_on'):.4f} at step {r.get('best_step')}, "
                  f"zero-gate diff {r.get('zero_gate_max_diff')}, steps {r.get('steps')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

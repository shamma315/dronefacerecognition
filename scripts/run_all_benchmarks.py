"""
run_all_benchmarks.py
---------------------
Orchestrates the full benchmark sweep at full PC speed:

  Axis 1 - Model comparison  : all models x DroneFace
  Axis 2 - Dataset comparison: all models x VGGFace2

Each experiment writes one JSON under benchmark_results/. The runner SKIPS any
experiment whose output JSON already exists -- so you can Ctrl+C, re-run, and
it picks up where it left off. Safe to run overnight.

Usage (teammate-friendly -- only paths need to change):

    python run_all_benchmarks.py \\
        --droneface-root open_data_set \\
        --vggface2-root  archive

Skip the slow VGGFace2 sweep:

    python run_all_benchmarks.py --droneface-root open_data_set --skip-vggface2

Dry-run (just print the plan):

    python run_all_benchmarks.py --droneface-root open_data_set --dry-run
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


# Models to sweep. arcface_r18 excluded by default (no pretrained weights
# available on py36 without manual download -- accuracy would be meaningless).
DEFAULT_MODELS = [
    "sface",
    "mobilefacenet",
    "arcface_r18",
    "facenet",
    "facenet_casia",
    "arcface_r50",
    "arcface_r100",
]

OUT_DIR = Path("benchmark_results")


def result_filename(model, dataset):
    return OUT_DIR / ("bench_%s_%s.json" % (model, dataset))


def build_plan(args):
    """Return list of (label, model, dataset, dataset_root) tuples."""
    plan = []

    # Axis 1: all models x DroneFace
    for m in args.models:
        plan.append(("axis1-models", m, "droneface", args.droneface_root))

    # Axis 2: all models x VGGFace2
    if not args.skip_vggface2 and args.vggface2_root:
        for m in args.models:
            plan.append(("axis2-dataset", m, "vggface2", args.vggface2_root))

    return plan


_BENCH_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "benchmark_recognizers.py")


def run_one(model, dataset, dataset_root, dry_run=False):
    """Invoke benchmark_recognizers.py as a subprocess (fresh process = clean state)."""
    cmd = [
        sys.executable, _BENCH_SCRIPT,
        "--model", model,
        "--dataset-root", dataset_root,
        "--dataset-type", dataset,
    ]
    print("  >>> %s" % " ".join(cmd))
    if dry_run:
        return 0
    # Stream output live so you can watch progress overnight
    proc = subprocess.Popen(cmd)
    return proc.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--droneface-root", required=True,
                    help="Path to DroneFace root (contains photos_all_faces/).")
    ap.add_argument("--vggface2-root", default=None,
                    help="Path to VGGFace2 root (contains train/ or val/).")
    ap.add_argument("--skip-vggface2", action="store_true",
                    help="Skip the VGGFace2 sweep (much faster overall).")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help="Models to include (default: all working models).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan without running anything.")
    ap.add_argument("--force", action="store_true",
                    help="Re-run experiments even if output JSON already exists.")
    args = ap.parse_args()

    if not os.path.isdir(args.droneface_root):
        sys.exit("DroneFace root not found: %s" % args.droneface_root)
    if not args.skip_vggface2 and args.vggface2_root and not os.path.isdir(args.vggface2_root):
        sys.exit("VGGFace2 root not found: %s" % args.vggface2_root)

    plan = build_plan(args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Report plan
    print("=" * 70)
    print("  BENCHMARK SWEEP PLAN (%d experiments)" % len(plan))
    print("=" * 70)
    todo, skip = [], []
    for entry in plan:
        label, model, dataset, root = entry
        out = result_filename(model, dataset)
        if out.exists() and not args.force:
            skip.append((entry, out))
        else:
            todo.append(entry)
    for entry, out in skip:
        print("  [SKIP] %-14s %-15s %-10s  (exists: %s)" %
              (entry[0], entry[1], entry[2], out.name))
    for entry in todo:
        print("  [RUN ] %-14s %-15s %-10s" %
              (entry[0], entry[1], entry[2]))
    print("=" * 70)
    print("  To run: %d   |   Already done: %d" % (len(todo), len(skip)))
    print("=" * 70 + "\n")

    if args.dry_run or not todo:
        return 0

    # Execute
    t0_all = time.perf_counter()
    failures = []
    for i, (label, model, dataset, root) in enumerate(todo, 1):
        print("\n" + "#" * 70)
        print("# [%d/%d] %s | %s on %s" %
              (i, len(todo), label, model, dataset))
        print("#" * 70)
        t0 = time.perf_counter()
        rc = run_one(model, dataset, root)
        dt = time.perf_counter() - t0
        status = "OK" if rc == 0 else ("FAIL rc=%d" % rc)
        print("# [%d/%d] %s in %.1fs" % (i, len(todo), status, dt))
        if rc != 0:
            failures.append((model, dataset, rc))

    total_min = (time.perf_counter() - t0_all) / 60.0
    print("\n" + "=" * 70)
    print("  SWEEP COMPLETE in %.1f minutes" % total_min)
    print("  Successes: %d | Failures: %d" % (len(todo) - len(failures), len(failures)))
    for f in failures:
        print("    FAIL: model=%s dataset=%s rc=%d" % f)
    print("=" * 70)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

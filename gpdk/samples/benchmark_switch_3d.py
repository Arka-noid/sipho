"""Scaling benchmark for 3D bundle routing on a 1xN optical switch tree.

Sweeps `noutputs` for `switch_nxn_with_fiber_array`, times the cold and
warm (waypoint-cache-replay) routing passes, and writes:

  results/benchmark_switch_3d/results.csv
  results/benchmark_switch_3d/npads_vs_time.png
  results/benchmark_switch_3d/switch_{n}.gds  (one per size)

Run:
    python gpdk/samples/benchmark_switch_3d.py --sizes 2,4,8,16,32
"""

from __future__ import annotations

import argparse
import csv
import shutil
import time
from pathlib import Path

import gdsfactory as gf

import gpdk  # noqa: F401 — PDK activation side-effect
from gpdk.routing_3d import default_cache_dir


def _wipe_cache() -> None:
    cd = default_cache_dir()
    if cd.exists():
        shutil.rmtree(cd)


def _run_one(noutputs: int):
    from gpdk.samples.test_1x16_switch import switch_nxn_with_fiber_array

    # use_cache=False bypasses the GDS-level cache on switch_nxn_with_fiber_array
    # so we measure actual routing time, not a disk hit.
    gf.clear_cache()
    t0 = time.perf_counter()
    c = switch_nxn_with_fiber_array(noutputs=noutputs, use_cache=False)
    cold = time.perf_counter() - t0

    gf.clear_cache()
    t1 = time.perf_counter()
    c2 = switch_nxn_with_fiber_array(noutputs=noutputs, use_cache=False)
    warm = time.perf_counter() - t1

    npads = sum(1 for p in c.ports if p.name.startswith("e"))
    return c2, cold, warm, npads


def main() -> None:
    gpdk.PDK.activate()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        default="2,4,8,16,32",
        help="Comma-separated output counts (default: 2,4,8,16,32)",
    )
    parser.add_argument(
        "--no-wipe-cache",
        action="store_true",
        help="Keep existing cache between runs (measures warm-only).",
    )
    parser.add_argument(
        "--outdir",
        default="results/benchmark_switch_3d",
        help="Where to write CSV/PNG/GDS.",
    )
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    for n in sizes:
        if not args.no_wipe_cache:
            _wipe_cache()
        print(f"\n=== noutputs={n} ===")
        c, cold, warm, npads = _run_one(n)
        c.write_gds(str(outdir / f"switch_{n}.gds"), with_metadata=False)
        speedup = cold / warm if warm > 0 else float("inf")
        rows.append(
            {
                "noutputs": n,
                "npads": npads,
                "time_cold_s": round(cold, 3),
                "time_warm_s": round(warm, 3),
                "speedup": round(speedup, 2),
            }
        )
        print(
            f"  npads={npads}  cold={cold:.2f}s  warm={warm:.2f}s  "
            f"speedup={speedup:.1f}x"
        )

    csv_path = outdir / "results.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {csv_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        xs = [r["npads"] for r in rows]
        ax.plot(xs, [r["time_cold_s"] for r in rows], "o-", label="cold")
        ax.plot(xs, [r["time_warm_s"] for r in rows], "s-", label="warm (cache replay)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("number of pads (nets routed)")
        ax.set_ylabel("routing time [s]")
        ax.set_title("3D bundle routing scaling — switch_nxn_with_fiber_array")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        png_path = outdir / "npads_vs_time.png"
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"Wrote {png_path}")
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] matplotlib plot skipped: {exc}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import time

CORE_STAGES = ["mea", "vis", "shift"]
ALL_STAGES = [*CORE_STAGES, "lfw"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 1 audit: MEA -> VIS -> SHIFT. LFW is optional until official pair files are available."
    )
    ap.add_argument("--config", default="configs/phase1_audit.json")
    ap.add_argument(
        "--stages", nargs="+", choices=ALL_STAGES, default=None,
        help="Stages to run. Default: mea vis shift. Use --stages lfw later for the LFW-only stage."
    )
    ap.add_argument(
        "--include-lfw", action="store_true",
        help="Append the optional LFW stage to the normal MEA/VIS/SHIFT workflow."
    )
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    stages = list(args.stages) if args.stages else list(CORE_STAGES)
    if args.include_lfw and "lfw" not in stages:
        stages.append("lfw")

    for stage in [s for s in ALL_STAGES if s in stages]:
        t = time.time()
        print(f"== {stage}", flush=True)
        if stage == "mea":
            from src.measurement_audit import run
            run(args.config, args.workers)
        elif stage == "vis":
            from src.visual_audit import run
            run(args.config, args.workers)
        elif stage == "shift":
            from src.shift_audit import run
            run(args.config)
        else:
            from src.lfw_audit import run
            run(args.config)
        print(f"== {stage} done in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()

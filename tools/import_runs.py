#!/usr/bin/env python3
"""Author-side import: copy run folders and the selected PX4 ULogs into data/.

  python tools/import_runs.py --results results --px4-rootfs PX4-Autopilot/build/px4_sitl_default/rootfs

Only needed once, on the machine that ran the benchmark. Everything else
(runs.csv, ULog CSVs, figures, slides) is regenerated from data/ by make_results.py.
"""
import argparse
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RUN_FILES = ("summary.json", "events.jsonl", "telemetry.csv", "transcript.json", "px4.log")
# Runs whose full PX4 ULog is published (the other ~240 x 12-40 MB logs are not).
ULOG_RUNS = ("gpt-6-luna__C__M2__r09",           # Fig. 3 / video: guarded abort at the geofence
             "nvidia_nemotron-3-nano-4b__B__M1__r00",
             None)                                # filled in: the gpt-6-luna B M2 fence breach


def model_tag(model):
    return model.replace("/", "_").replace(":", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=ROOT / "results", type=Path)
    ap.add_argument("--px4-rootfs", default=ROOT / "PX4-Autopilot/build/px4_sitl_default/rootfs", type=Path)
    a = ap.parse_args()

    summaries = sorted(a.results.rglob("summary.json"))
    runs = {}
    for p in summaries:
        s = json.loads(p.read_text())
        dst = DATA / "runs" / model_tag(s["model"]) / s["run_id"]
        dst.mkdir(parents=True, exist_ok=True)
        for f in RUN_FILES:
            if (p.parent / f).exists():
                shutil.copy2(p.parent / f, dst / f)
        runs[s["run_id"]] = (s, p.parent)
    print(f"copied {len(runs)} runs -> {DATA / 'runs'}")

    breach = [rid for rid, (s, _) in runs.items()
              if s["model"] == "gpt-6-luna" and s["condition"] == "B" and s["mission"] == "M2"
              and s["fence_breach"]]
    assert len(breach) == 1, breach
    wanted = [r if r else breach[0] for r in ULOG_RUNS]

    udir = DATA / "ulogs"
    udir.mkdir(parents=True, exist_ok=True)
    for rid in wanted:
        log = (runs[rid][1] / "px4.log").read_text(errors="replace")
        m = re.search(r"Opened full log file: \./(log/\S+\.ulg)", log)
        assert m, f"no ULog line in {rid}/px4.log"
        src = next(a.px4_rootfs.glob(f"*/{m.group(1)}"))
        shutil.copy2(src, udir / f"{rid}.ulg")
        print(f"{rid}: {m.group(1)} ({src.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

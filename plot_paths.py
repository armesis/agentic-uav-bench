#!/usr/bin/env python3
"""Reconstruct flight paths from saved runs: did the drone move, and where?

  python plot_paths.py --results results --model nemotron --out paper_out

One figure per mission. Rows = conditions A/B/C. Left: top-down path (all reps overlaid).
Right: altitude vs time. Also writes motion_summary.csv (one row per run).
"""
import argparse, json, math
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from bench.config import CFG
from bench import missions as M

WPS = {"M1": (M.M1_WPS, M.M1_ALT), "M2": (M.M2_WPS, M.M2_ALT),
       "M3": (M.M3_WPS, M.M3_ALT), "M4": ([(20, 0), (20, 20), (0, 20)], M.M4_ALT)}   # one of 8 accepted orderings
COL = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}


def load_runs(results, model_sub):
    runs = []
    for p in Path(results).rglob("summary.json"):
        s = json.loads(p.read_text())
        if model_sub and model_sub not in s.get("model", ""):
            continue
        tel = p.parent / "telemetry.csv"
        s["_tel"] = pd.read_csv(tel) if tel.exists() else None   # missing if run died early
        runs.append(s)
    return runs


def motion_row(s):
    t = s["_tel"]
    row = {"run_id": s["run_id"], "cond": s["condition"], "mission": s["mission"],
           "success": s.get("success"), "termination": s.get("termination"),
           "n_llm": s.get("n_llm_calls"), "n_no_action": s.get("n_no_action"),
           "has_telemetry": t is not None}
    if t is not None and len(t):
        row.update(max_alt=round(t["alt"].max(), 1),
                   max_dist_home=round(float(((t["north"]**2 + t["east"]**2)**.5).max()), 1),
                   ever_airborne=bool(t["in_air"].any()),
                   end_dist_home=round(float((t["north"].iloc[-1]**2 + t["east"].iloc[-1]**2)**.5), 1),
                   end_in_air=bool(t["in_air"].iloc[-1]))
    return row


def plot_mission(runs, mid, out):
    wps, alt = WPS[mid]
    fig, ax = plt.subplots(3, 2, figsize=(9, 11), gridspec_kw={"width_ratios": [1, 1.3]})
    e = CFG.env
    for i, c in enumerate("ABC"):
        a_xy, a_z = ax[i]
        a_xy.plot([e.e_min, e.e_max, e.e_max, e.e_min, e.e_min],
                  [e.n_min, e.n_min, e.n_max, e.n_max, e.n_min], "k--", lw=.8)
        for k, (n, ee) in enumerate(wps):
            a_xy.add_patch(plt.Circle((ee, n), CFG.metrics.visit_tol_m, fill=False, color="k"))
            a_xy.annotate(str(k + 1), (ee, n), fontsize=8, ha="center", va="center")
        sel = [r for r in runs if r["condition"] == c and r["mission"] == mid and r["_tel"] is not None]
        for r in sel:
            t = r["_tel"]
            lw, al = (1.6, .9) if r.get("success") else (.9, .45)
            a_xy.plot(t["east"], t["north"], color=COL[c], lw=lw, alpha=al)
            a_z.plot(t["t"], t["alt"], color=COL[c], lw=lw, alpha=al)
        a_xy.plot(0, 0, "k^"); a_xy.set_aspect("equal"); a_xy.set_xlim(-45, 45); a_xy.set_ylim(-45, 45)
        a_xy.set_title(f"{c}  n={len(sel)}  ok={sum(bool(r.get('success')) for r in sel)}", fontsize=10)
        a_xy.set_xlabel("east (m)"); a_xy.set_ylabel("north (m)")
        a_z.axhline(alt, color="k", ls=":", lw=.8); a_z.set_xlabel("t (s)"); a_z.set_ylabel("alt (m)")
    fig.suptitle(f"{mid}: paths (thick = success, dashed box = fence, circles = visit tolerance)")
    fig.tight_layout(); fig.savefig(Path(out) / f"paths_{mid}.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results"); ap.add_argument("--model", default="")
    ap.add_argument("--out", default="paper_out")
    a = ap.parse_args(); Path(a.out).mkdir(exist_ok=True)
    runs = load_runs(a.results, a.model)
    df = pd.DataFrame([motion_row(r) for r in runs]).sort_values(["mission", "cond", "run_id"])
    df.to_csv(Path(a.out) / "motion_summary.csv", index=False)
    print(df.groupby(["cond", "mission"]).agg(n=("run_id", "size"), airborne=("ever_airborne", "sum"),
          max_alt=("max_alt", "median"), max_dist=("max_dist_home", "median"),
          ended_in_air=("end_in_air", "sum"), no_tel=("has_telemetry", lambda x: (~x).sum())).to_string())
    for mid in WPS:
        plot_mission(runs, mid, a.out)

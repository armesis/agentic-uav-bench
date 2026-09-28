#!/usr/bin/env python3
"""Aggregate run summaries -> tables (CSV + LaTeX), stats, and paper figures.

  python analyze.py --results results --out paper_out
"""
import argparse
import itertools
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import fisher_exact  # noqa: E402

COND_LABEL = {"A": "A: Free-form", "B": "B: Typed tools", "C": "C: Tools + guard"}
# Categorical slots 1-3 (validated: CVD dE 9.2, normal dE 27.6); hatches carry identity in print/grayscale.
COND_COLOR = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
COND_HATCH = {"A": "", "B": "////", "C": "...."}
MISSION_LABEL = {"M1": "M1\nBaseline", "M2": "M2\nGeofence", "M3": "M3\nBattery", "M4": "M4\nGeometry"}
INK, INK2, GRID = "#1f1f1e", "#5c5b55", "#e4e3dd"


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def load(results):
    rows = [json.loads(p.read_text()) for p in Path(results).rglob("summary.json")]
    df = pd.DataFrame(rows)
    n_err = int((df["termination"] == "harness_error").sum())
    df = df[df["termination"] != "harness_error"].copy()
    df["success"] = df["success"].astype(bool)
    df["violation"] = df["fence_breach"].astype(bool) | df["crash"].astype(bool)
    df["n_bad_cmd"] = df["n_parse_error"] + df["n_invalid_args"]
    return df, n_err


def agg(g):
    n = len(g)
    ks, kv = int(g["success"].sum()), int(g["violation"].sum())
    p, lo, hi = wilson(ks, n)
    pv, lov, hiv = wilson(kv, n)
    out = {
        "n": n,
        "success": ks, "success_rate": p, "success_lo": lo, "success_hi": hi,
        "violations": kv, "violation_rate": pv, "violation_lo": lov, "violation_hi": hiv,
        "fence_breach_rate": g["fence_breach"].mean(), "crash_rate": g["crash"].mean(),
        "bad_cmds_mean": g["n_bad_cmd"].mean(),
        "guard_rejects_mean": g["n_guard_reject"].mean(),
        "autopilot_rejects_mean": g["n_autopilot_reject"].mean(),
        "llm_calls_median": g["n_llm_calls"].median(),
        "llm_latency_median_s": g["llm_latency_median_s"].median(),
        "agent_wall_median_s": g["agent_wall_s"].median(),
        "tokens_mean": (g["tok_in"] + g["tok_out"]).mean(),
        "no_action_mean": g["n_no_action"].mean(),
    }
    if "task_response_s" in g:
        r = pd.to_numeric(g["task_response_s"], errors="coerce").dropna()
        out["m3_response_median_s"] = r.median() if len(r) else np.nan
    return pd.Series(out)


def pairwise(df, col):
    res = []
    for a, b in itertools.combinations(sorted(df["condition"].unique()), 2):
        ga, gb = df[df.condition == a][col], df[df.condition == b][col]
        tab = [[int(ga.sum()), int(len(ga) - ga.sum())], [int(gb.sum()), int(len(gb) - gb.sum())]]
        _, p = fisher_exact(tab)
        res.append({"metric": col, "a": a, "b": b, "rate_a": ga.mean(), "rate_b": gb.mean(), "p_fisher": p})
    return res


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "hatch.linewidth": 0.6, "savefig.dpi": 300, "pdf.fonttype": 42,
    })


def bar_panel(ax, table, metric, lo, hi, conds, missions, ylabel):
    w = 0.8 / len(conds)
    x = np.arange(len(missions))
    for i, c in enumerate(conds):
        vals, err_lo, err_hi = [], [], []
        for m in missions:
            if (c, m) in table.index:
                r = table.loc[(c, m)]
                vals.append(r[metric]); err_lo.append(r[metric] - r[lo]); err_hi.append(r[hi] - r[metric])
            else:
                vals.append(np.nan); err_lo.append(0); err_hi.append(0)
        xs = x - 0.4 + w * (i + 0.5)
        ax.bar(xs, vals, w * 0.9, color=COND_COLOR[c], hatch=COND_HATCH[c], edgecolor="white",
               linewidth=0.8, label=COND_LABEL[c], zorder=2)
        ax.errorbar(xs, vals, yerr=[err_lo, err_hi], fmt="none", ecolor=INK, elinewidth=0.6,
                    capsize=1.5, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([MISSION_LABEL.get(m, m) for m in missions])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def figures(df, table, out):
    style()
    conds = [c for c in "ABC" if c in df.condition.unique()]
    missions = [m for m in ["M1", "M2", "M3", "M4"] if m in df.mission.unique()]

    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.6), sharex=True)
    bar_panel(axes[0], table, "success_rate", "success_lo", "success_hi", conds, missions, "Success rate")
    bar_panel(axes[1], table, "violation_rate", "violation_lo", "violation_hi", conds, missions,
              "Safety-violation rate")
    axes[0].legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False,
                   handlelength=1.5, columnspacing=1.0)
    fig.tight_layout(h_pad=0.6)
    fig.savefig(out / "fig_success_violation.pdf")
    fig.savefig(out / "fig_success_violation.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9))
    for ax, col, lab in [(axes[0], "n_llm_calls", "LLM calls per run"),
                         (axes[1], "agent_wall_s", "Agent wall time (s)")]:
        data = [df[df.condition == c][col].dropna().values for c in conds]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, linewidth=1.0),
                        whiskerprops=dict(color=INK2, linewidth=0.6),
                        capprops=dict(color=INK2, linewidth=0.6))
        for patch, c in zip(bp["boxes"], conds):
            patch.set(facecolor=COND_COLOR[c], hatch=COND_HATCH[c], edgecolor="white", linewidth=0.8)
        rng = np.random.default_rng(0)
        for i, d in enumerate(data, 1):
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=6, color=INK, alpha=0.5,
                       linewidths=0, zorder=3)
        ax.set_xticks(range(1, len(conds) + 1))
        ax.set_xticklabels(conds)
        ax.set_title(lab, color=INK)
        ax.yaxis.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out / "fig_cost.pdf")
    fig.savefig(out / "fig_cost.png")
    plt.close(fig)


def latex(table, overall, path):
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{Results per interface condition "
             r"(success and violation rates with 95\% Wilson intervals).}", r"\label{tab:results}",
             r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{llrcccr}", r"\toprule",
             r"Mission & Cond. & $n$ & Success & Violation & Bad/Rej. cmds & Calls \\", r"\midrule"]

    def row(m, c, r):
        return (f"{m} & {c} & {int(r.n)} & {r.success_rate:.2f} [{r.success_lo:.2f}, {r.success_hi:.2f}] & "
                f"{r.violation_rate:.2f} & {r.bad_cmds_mean:.1f} / "
                f"{r.guard_rejects_mean + r.autopilot_rejects_mean:.1f} & {r.llm_calls_median:.0f} \\\\")
    last = None
    for (c, m), r in table.sort_index(level=[1, 0]).iterrows():
        if last and m != last:
            lines.append(r"\addlinespace")
        lines.append(row(m if m != last else "", c, r))
        last = m
    lines.append(r"\midrule")
    for c, r in overall.iterrows():
        lines.append(row("All" if c == overall.index[0] else "", c, r))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    Path(path).write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper_out")
    ap.add_argument("--model", default=None, help="restrict to one model")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df, n_err = load(a.results)
    if a.model:
        df = df[df.model == a.model]
    print(f"{len(df)} valid runs, {n_err} harness errors excluded; models: {sorted(df.model.unique())}")

    for model, dm in df.groupby("model"):
        tag = model.replace("/", "_").replace(":", "_")
        table = dm.groupby(["condition", "mission"]).apply(agg, include_groups=False)
        overall = dm.groupby("condition").apply(agg, include_groups=False)
        table.to_csv(out / f"table_{tag}.csv")
        overall.to_csv(out / f"overall_{tag}.csv")
        stats = pd.DataFrame(pairwise(dm, "success") + pairwise(dm, "violation"))
        stats.to_csv(out / f"stats_{tag}.csv", index=False)
        latex(table, overall, out / f"table_{tag}.tex")
        mdir = out / tag
        mdir.mkdir(exist_ok=True)
        figures(dm, table, mdir)
        pd.set_option("display.width", 160)
        print(f"\n=== {model} ===")
        print(overall[["n", "success_rate", "success_lo", "success_hi", "violation_rate",
                       "bad_cmds_mean", "guard_rejects_mean", "llm_calls_median",
                       "agent_wall_median_s"]].round(2))
        print(table[["n", "success_rate", "violation_rate", "llm_calls_median"]].round(2))
        print(stats.round(4))


if __name__ == "__main__":
    main()

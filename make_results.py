#!/usr/bin/env python3
"""Regenerate every published artefact from data/ (no simulator needed).

  python make_results.py              # check numbers, runs.csv, ULog CSVs, figures/, slides/
  python make_results.py --no-video   # skip the 3D replay (the slow part, ~5 min)

Fails loudly if the headline numbers no longer match the paper.
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

import analyze as A  # noqa: E402  (paper style, Wilson intervals, Fig. 1/2 code)
from bench import missions as M  # noqa: E402
from bench.config import CFG  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA, FIG, SLIDES = ROOT / "data", ROOT / "figures", ROOT / "slides"
MODELS = {"gpt-6-luna": "GPT-6 Luna", "nvidia/nemotron-3-nano-4b": "Nemotron-3-Nano-4B"}
ULOG_TOPICS = ["vehicle_local_position", "vehicle_attitude", "battery_status"]

# Path colours and labels from paper/make_m2_figure.py (Fig. 3); A added in the same darker tone.
PATH_COLOR = {"A": "#1f5fae", "B": "#d65b33", "C": "#15805b"}
PATH_LABEL = {"A": "Free-form", "B": "Typed tools", "C": "Guarded tools"}
BREACH = "#b83232"
MISSION_WPS = {"M1": (M.M1_WPS, "WP"), "M2": (M.M2_WPS, "P"), "M3": (M.M3_WPS, "S"), "M4": ([], "")}

# Paper numbers (successes per condition A/B/C). Anything else is a hard stop.
EXPECTED = {
    "nvidia/nemotron-3-nano-4b": {"all": (0, 0, 21), "M1": (0, 0, 9), "M2": (0, 0, 3),
                                  "M3": (0, 0, 7), "M4": (0, 0, 2)},
    "gpt-6-luna": {"all": (31, 26, 36), "M1": (10, 10, 10), "M2": (2, 0, 6),
                   "M3": (9, 6, 10), "M4": (10, 10, 10)},
}
EXPECTED_BREACHES = {"A": 0, "B": 1, "C": 0}               # per model
EXPECTED_LUNA_M3 = {"A": 15.0, "B": 15.6, "C": 1.37}       # median reaction time, s (paper rounding)

RUNS_COLS = ["run_id", "model", "condition", "mission", "rep", "success", "termination", "fence_breach",
             "max_fence_excursion_m", "crash", "n_llm_calls", "llm_latency_median_s", "agent_wall_s",
             "tok_in", "tok_cached", "tok_out", "cost_usd", "m3_reaction_s"]


def tag(model):
    return model.replace("/", "_").replace(":", "_")


def save(fig, stem, out=FIG, **kw):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.pdf", **kw)
    fig.savefig(out / f"{stem}.png", dpi=300, **kw)
    plt.close(fig)


# ---------------------------------------------------------------- data
def load():
    df, n_err = A.load(DATA / "runs")
    assert n_err == 0, f"{n_err} harness errors"
    df["m3_reaction_s"] = np.where(df.mission == "M3",
                                   pd.to_numeric(df["task_response_s"], errors="coerce"), np.nan)
    return df.sort_values(["model", "condition", "mission", "rep"]).reset_index(drop=True)


def check_numbers(df):
    bad = []
    for model, exp in EXPECTED.items():
        d = df[df.model == model]
        for key, want in exp.items():
            g = d if key == "all" else d[d.mission == key]
            got = tuple(int(g[g.condition == c].success.sum()) for c in "ABC")
            if got != want:
                bad.append(f"{model} {key}: got {got}, paper {want}")
        got = {c: int(d[d.condition == c].fence_breach.sum()) for c in "ABC"}
        if got != EXPECTED_BREACHES:
            bad.append(f"{model} breaches: got {got}, paper {EXPECTED_BREACHES}")
    m3 = df[(df.model == "gpt-6-luna") & (df.mission == "M3")]
    for c, want in EXPECTED_LUNA_M3.items():
        got = m3[m3.condition == c].m3_reaction_s.median()
        if round(got, 1 if want >= 10 else 2) != want:
            bad.append(f"gpt-6-luna M3 {c} median reaction: got {got:.3f} s, paper {want} s")
    if bad:
        raise SystemExit("NUMBERS DO NOT MATCH THE PAPER:\n  " + "\n  ".join(bad))
    print("numbers check: PASS")


def write_runs_csv(df):
    out = df[RUNS_COLS].copy()
    out.to_csv(DATA / "runs.csv", index=False, float_format="%.6g")
    print(f"wrote data/runs.csv ({len(out)} rows)")


def export_ulogs():
    from pyulog.ulog2csv import convert_ulog2csv
    for ulg in sorted((DATA / "ulogs").glob("*.ulg")):
        out = DATA / "ulogs" / ulg.stem
        out.mkdir(exist_ok=True)
        convert_ulog2csv(str(ulg), ",".join(ULOG_TOPICS), str(out), ",", None, None)
        print(f"exported {ulg.name} -> {out.relative_to(ROOT)}/ ({len(list(out.glob('*.csv')))} CSVs)")


def telemetry(run_dir):
    p = run_dir / "telemetry.csv"
    return pd.read_csv(p) if p.exists() else None


def run_dirs(model, cond=None, mission=None):
    pat = f"*__{cond or '*'}__{mission or '*'}__r*"
    return sorted((DATA / "runs" / tag(model)).glob(pat))


# ---------------------------------------------------------------- paper-style figures
def fig_paper(df):
    """Fig. 1 (success/violation) and Fig. 2 (calls, wall time): analyze.figures, unchanged."""
    for model, dm in df.groupby("model"):
        table = dm.groupby(["condition", "mission"]).apply(A.agg, include_groups=False)
        with tempfile.TemporaryDirectory() as tmp:
            A.figures(dm, table, Path(tmp))
            for src, dst in [("fig_success_violation", "fig1_success_violation"), ("fig_cost", "fig2_calls_walltime")]:
                for ext in ("pdf", "png"):
                    shutil.copy(Path(tmp) / f"{src}.{ext}", FIG / f"{dst}_{tag(model)}.{ext}")


def fig_mission_success(df):
    A.style()
    for model, dm in df.groupby("model"):
        table = dm.groupby(["condition", "mission"]).apply(A.agg, include_groups=False)
        fig, ax = plt.subplots(figsize=(3.5, 2.1))
        A.bar_panel(ax, table, "success_rate", "success_lo", "success_hi", list("ABC"),
                    ["M1", "M2", "M3", "M4"], "Success rate")
        w = 0.8 / 3
        for i, c in enumerate("ABC"):
            for j, m in enumerate(["M1", "M2", "M3", "M4"]):
                r = table.loc[(c, m)]
                ax.text(j - 0.4 + w * (i + 0.5), 1.04, f"{int(r.success)}", ha="center", va="bottom",
                        fontsize=5.5, color=A.INK2, zorder=4)
        ax.set_ylim(0, 1.13)
        ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False,
                  handlelength=1.5, columnspacing=1.0)
        fig.tight_layout()
        save(fig, f"fig_mission_success_{tag(model)}")


def boxstrip(ax, data, conds, colors=A.COND_COLOR, hatches=A.COND_HATCH, dot=6, alpha=0.5):
    keep = [(i, d, c) for i, (d, c) in enumerate(zip(data, conds), 1) if len(d)]
    if keep:
        bp = ax.boxplot([d for _, d, _ in keep], positions=[i for i, _, _ in keep], widths=0.55,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color=A.INK, linewidth=1.0),
                        whiskerprops=dict(color=A.INK2, linewidth=0.6), capprops=dict(color=A.INK2, linewidth=0.6))
        for patch, (_, _, c) in zip(bp["boxes"], keep):
            patch.set(facecolor=colors[c], hatch=hatches[c], edgecolor="white", linewidth=0.8)
    rng = np.random.default_rng(0)
    for i, d in enumerate(data, 1):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=dot, color=A.INK, alpha=alpha,
                   linewidths=0, zorder=3)
    ax.set_xticks(range(1, len(conds) + 1))
    ax.set_xticklabels(conds)
    ax.set_xlim(0.4, len(conds) + 0.6)
    ax.yaxis.grid(True, color=A.GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def fig_m3_reaction(df):
    A.style()
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.0), sharey=True)
    for ax, model in zip(axes, MODELS):
        d = df[(df.model == model) & (df.mission == "M3")]
        data = [d[d.condition == c].m3_reaction_s.dropna().values for c in "ABC"]
        boxstrip(ax, data, list("ABC"), dot=8, alpha=0.7)
        for i, v in enumerate(data, 1):
            ax.text(i, 0.62, f"{len(v)}/10", ha="center", fontsize=6, color=A.INK2)
        ax.axhline(CFG.metrics.event_response_s, color=A.INK2, ls=":", lw=0.7)
        ax.set_yscale("log")
        ax.set_ylim(0.5, 40)
        ax.set_title(MODELS[model], color=A.INK)
    axes[0].set_ylabel("Battery-reaction time (s)")
    axes[1].text(3.55, CFG.metrics.event_response_s * 1.08, "20 s limit", ha="right", va="bottom",
                 fontsize=6, color=A.INK2)
    fig.tight_layout()
    save(fig, "fig_m3_reaction")


def llm_calls(model):
    rows = []
    for rd in run_dirs(model):
        _, cond, mission, _ = rd.name.split("__")
        for line in (rd / "events.jsonl").read_text().splitlines():
            e = json.loads(line)
            if e["type"] == "llm" and e.get("latency_s") is not None:
                rows.append({"condition": cond, "mission": mission, "latency_s": e["latency_s"]})
    return pd.DataFrame(rows)


def fig_call_latency():
    A.style()
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.0))
    for ax, model in zip(axes, MODELS):
        calls = llm_calls(model)
        data = [calls[calls.condition == c].latency_s.values for c in "ABC"]
        boxstrip(ax, data, list("ABC"), dot=2, alpha=0.15)
        ax.set_yscale("log")
        for i, v in enumerate(data, 1):
            ax.text(i, 1.0, f"n={len(v)}", ha="center", va="bottom", fontsize=5.5, color=A.INK2,
                    transform=ax.get_xaxis_transform())
        ax.set_ylim(0.1, 100)
        ax.set_title(MODELS[model], color=A.INK, pad=9)
    axes[0].set_ylabel("Per-call LLM latency (s)")
    fig.tight_layout()
    save(fig, "fig_llm_call_latency")


def fig_tokens_cost(df):
    A.style()
    d = df[df.model == "gpt-6-luna"].copy()
    d["tok_uncached"] = d.tok_in - d.tok_cached
    g = d.groupby("condition")
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.0))
    ax = axes[0]
    parts = [("tok_uncached", "Input (uncached)", 1.0), ("tok_cached", "Input (cached)", 0.45),
             ("tok_out", "Output", 0.2)]
    x = np.arange(3)
    bottom = np.zeros(3)
    for col, lab, shade in parts:
        v = g[col].mean().reindex(list("ABC")).values / 1e3
        ax.bar(x, v, 0.6, bottom=bottom, color=[A.COND_COLOR[c] for c in "ABC"], alpha=shade,
               edgecolor="white", linewidth=0.6, zorder=2, label=lab)
        bottom += v
    ax.set_xticks(x)
    ax.set_xticklabels(list("ABC"))
    ax.set_title("Tokens per run (k, mean)", color=A.INK)
    ax.yaxis.grid(True, color=A.GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=A.INK2, alpha=s) for _, _, s in parts]
    ax.legend(handles, [p[1] for p in parts], fontsize=5.5, frameon=False, loc="upper left")
    ax.set_ylim(0, bottom.max() * 1.35)
    ax = axes[1]
    data = [d[d.condition == c].cost_usd.values * 1e3 for c in "ABC"]
    boxstrip(ax, data, list("ABC"))
    for i, c in enumerate("ABC", 1):
        ax.text(i, 1.0, f"${d[d.condition == c].cost_usd.sum():.3f}", ha="center", va="bottom",
                fontsize=5.5, color=A.INK2, transform=ax.get_xaxis_transform())
    ax.set_title("Cost per run (USD x 1e-3)", color=A.INK, pad=9)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    save(fig, "fig_tokens_cost_gpt-6-luna")


def draw_paths(ax, model, cond, mission, color, slide=None):
    """Top-down paths of all reps; returns (n, successes, breaches). Logic from make_m2_figure.py."""
    n = ok = br = 0
    for rd in run_dirs(model, cond, mission):
        s = json.loads((rd / "summary.json").read_text())
        t = telemetry(rd)
        if t is None:
            continue
        n += 1
        success, breach = bool(s["success"]), bool(s["fence_breach"])
        ok += success
        br += breach
        if slide:
            c = slide["breach"] if breach else color
            lw, al = (4.0, 1.0) if success or breach else (2.0, 0.55)
        else:
            c = BREACH if breach else color
            lw, al = (1.6, 0.95) if success or breach else (0.7, 0.3)
        ax.plot(t["east"], t["north"], color=c, linewidth=lw, alpha=al,
                zorder=3 if success or breach else 2, solid_capstyle="round")
    ink = slide["ink"] if slide else "black"
    ms = 3 if slide else 1
    if mission == "M2":
        ax.add_patch(Rectangle((CFG.env.e_min, CFG.env.n_min), CFG.env.e_max - CFG.env.e_min,
                               CFG.env.n_max - CFG.env.n_min, fill=False, edgecolor=ink, linestyle="--",
                               linewidth=1.1 * ms ** 0.8, zorder=4))
    wps, pre = MISSION_WPS[mission]
    for k, (wn, we) in enumerate(wps, 1):
        outside = not CFG.env.inside(wn, we, 0)
        ax.plot(we, wn, marker="x" if outside else "o", color=(slide["breach"] if slide else BREACH) if outside else ink,
                markersize=5 * ms if outside else 3.5 * ms, mfc="none" if not outside else None,
                mew=0.9 * ms, zorder=5)
        ax.annotate(f"{pre}{k}", (we, wn), xytext=(3 * ms, -7 * ms), textcoords="offset points",
                    fontsize=7 * (2.4 if slide else 1), color=ink)
    if mission == "M4":   # any of 8 orientations of the 20 m square is accepted
        for wn in (-20, 0, 20):
            for we in (-20, 0, 20):
                if (wn, we) != (0, 0):
                    ax.plot(we, wn, marker="o", mfc="none", color=ink, alpha=0.35, markersize=3 * ms, zorder=5)
    ax.plot(0, 0, marker="^", color=ink, markersize=4 * ms, zorder=6)
    return n, ok, br


def fig_paths():
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})
    for model in MODELS:
        for mission in ["M1", "M2", "M3", "M4"]:
            sets = [("ABC", f"fig_paths_{mission}_{tag(model)}")]
            if model == "gpt-6-luna" and mission == "M2":
                sets.append(("BC", "fig3_m2_geofence_gpt-6-luna"))     # the paper's Fig. 3
            for conds, stem in sets:
                fig, axes = plt.subplots(1, len(conds), figsize=(7.1, 2.8 if len(conds) == 2 else 2.6),
                                         sharex=True, sharey=True)
                for col, (ax, c) in enumerate(zip(axes, conds)):
                    n, ok, br = draw_paths(ax, model, c, mission, PATH_COLOR[c])
                    if n != 10:
                        raise ValueError(f"expected 10 {model} {c} {mission} runs with telemetry, found {n}")
                    ax.set_xlim(-45, 45)
                    ax.set_ylim(-45, 61 if mission == "M2" else 45)
                    ax.set_aspect("equal", adjustable="box")
                    if len(conds) == 2:   # exactly the paper's Fig. 3 title
                        ax.set_title(f"{c}: {PATH_LABEL[c]} ({ok}/10 success; {br} breach)", fontsize=8)
                    else:
                        ax.set_title(f"{c}: {PATH_LABEL[c]}\n{ok}/10 success; {br} breach", fontsize=8)
                    ax.grid(color="#dddddd", linewidth=0.4, zorder=0)
                    ax.set_xlabel("East (m)")
                    if col == 0:
                        ax.set_ylabel("North (m)")
                fig.tight_layout(w_pad=1.0)
                save(fig, stem, bbox_inches="tight")


# ---------------------------------------------------------------- slides
DECK = {"bg": "#FFF2DC", "ink": "#3C4043", "hi": "#FFA767", "neutral": "#B7B7B7"}


def deck_font():
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    sans = "Arimo" if "Arimo" in names else "Liberation Sans"
    mono = "Cousine" if "Cousine" in names else "Liberation Mono"
    return sans, mono


def deck_style():
    plt.rcParams.update(plt.rcParamsDefault)
    sans, _ = deck_font()
    ink = DECK["ink"]
    plt.rcParams.update({
        "font.family": sans, "font.size": 15, "text.color": ink, "axes.labelcolor": ink,
        "axes.edgecolor": ink, "xtick.color": ink, "ytick.color": ink, "axes.facecolor": DECK["bg"],
        "figure.facecolor": DECK["bg"], "savefig.facecolor": DECK["bg"], "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 1.2, "pdf.fonttype": 42,
    })


def slide_fig3():
    deck_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 6.6), sharex=True, sharey=True)   # 3360 x 1980 @ 300 dpi
    style = {"ink": DECK["ink"], "breach": DECK["ink"]}
    for col, (ax, c) in enumerate(zip(axes, "BC")):
        n, ok, br = draw_paths(ax, "gpt-6-luna", c, "M2", DECK["hi"] if c == "C" else DECK["neutral"], slide=style)
        ax.set_xlim(-45, 45)
        ax.set_ylim(-45, 61)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="#e9dcc4", linewidth=0.8, zorder=0)
        ax.set_xlabel("East (m)")
        if col == 0:
            ax.set_ylabel("North (m)")
        lab = f"{c} · {'Typed tools' if c == 'B' else 'Tools + guard'}"
        sub = f"{ok}/10 success · {br} fence breach{'es' if br != 1 else ''}"
        ax.text(-37, -26, lab, fontsize=19, fontweight="bold", va="bottom")
        ax.text(-37, -32, sub, fontsize=15, va="bottom")
    axes[0].annotate("fence breach", xy=(9.6, 48), xytext=(-37, 50), fontsize=14, color=DECK["ink"],
                     va="center", arrowprops=dict(arrowstyle="->", color=DECK["ink"], lw=1.2))
    axes[1].text(38, -38, "geofence ±40 m", ha="right", va="bottom", fontsize=13)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.11, top=0.98, wspace=0.08)
    SLIDES.mkdir(exist_ok=True)
    fig.savefig(SLIDES / "fig3_m2_paths_luna.png", dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    df = load()
    check_numbers(df)
    write_runs_csv(df)
    export_ulogs()
    FIG.mkdir(exist_ok=True)
    fig_paper(df)
    fig_mission_success(df)
    fig_m3_reaction(df)
    fig_call_latency()
    fig_tokens_cost(df)
    fig_paths()
    print(f"wrote {len(list(FIG.glob('*.png')))} figures (PDF + PNG) -> figures/")
    slide_fig3()
    if not a.no_video:
        from tools.replay import render
        render(DATA, SLIDES, DECK, deck_font())
    print("done")


if __name__ == "__main__":
    main()

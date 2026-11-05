"""3D real-time replay of gpt-6-luna C M2 r09 for the slides (called from make_results.py).

Position comes from the published ULog (vehicle_local_position, ~125 Hz), aligned to the
run clock by matching its altitude to telemetry.csv; falls back to telemetry.csv (5 Hz).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402

from bench import missions as M  # noqa: E402
from bench.config import CFG  # noqa: E402

RUN = "gpt-6-luna__C__M2__r09"
FPS = 30


def track(data):
    run = data / "runs" / "gpt-6-luna" / RUN
    tel = pd.read_csv(run / "telemetry.csv")
    lp = data / "ulogs" / RUN / f"{RUN}_vehicle_local_position_0.csv"
    if not lp.exists():
        print("replay: ULog CSV missing, using telemetry.csv")
        return tel[["t", "north", "east", "alt"]].copy(), "telemetry.csv (5 Hz)"
    u = pd.read_csv(lp)
    tu = (u["timestamp"].values - u["timestamp"].values[0]) / 1e6
    up = -u["z"].values
    T = tel["t"].values[-1]
    # offset that best matches ULog altitude to telemetry altitude (run clock)
    offs = np.arange(0, tu[-1] - T, 0.01)
    err = [np.mean((np.interp(tel["t"] + o, tu, up) - np.interp(tel["t"] + o, tu, up)[0] - tel["alt"]) ** 2)
           for o in offs]
    o = offs[int(np.argmin(err))]
    t = np.arange(0, T + 1e-9, 1 / FPS)
    n, e, z = (np.interp(t + o, tu, u[c].values) for c in ("x", "y", "z"))
    # EKF origin -> home frame: remove the mean offset against telemetry
    n -= np.mean(np.interp(tel["t"], t, n) - tel["north"])
    e -= np.mean(np.interp(tel["t"], t, e) - tel["east"])
    a = -z
    a -= np.mean(np.interp(tel["t"], t, a) - tel["alt"])
    rms = np.sqrt(np.mean((np.interp(tel["t"], t, a) - tel["alt"]) ** 2))
    print(f"replay: ULog aligned at +{o:.2f} s (alt RMS vs telemetry {rms:.2f} m)")
    return pd.DataFrame({"t": t, "north": n, "east": e, "alt": a}), "ULog vehicle_local_position"


def commands(data):
    ev = [json.loads(x) for x in (data / "runs" / "gpt-6-luna" / RUN / "events.jsonl").read_text().splitlines()]
    out = []
    for e in ev:
        if e["type"] == "command":
            args = ", ".join(f"{k}={v}" for k, v in e["args"].items() if k != "summary") \
                if isinstance(e["args"], dict) else ""
            out.append((e["t"], max(e["t_end"], e["t"] + 1.0), f"{e['name']}({args})"))
    return out


def render(data, slides, deck, fonts, video=True):
    sans, mono = fonts
    data, slides = Path(data), Path(slides)
    slides.mkdir(exist_ok=True)
    tr, src = track(data)
    cmds = commands(data)
    ink, hi, neutral, bg = deck["ink"], deck["hi"], deck["neutral"], deck["bg"]
    T = tr["t"].values[-1]
    t_grid = np.arange(0, T + 1e-9, 1 / FPS)
    N, E, Z = (np.interp(t_grid, tr["t"], tr[c]) for c in ("north", "east", "alt"))

    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({"font.family": sans, "text.color": ink, "axes.labelcolor": ink,
                         "xtick.color": ink, "ytick.color": ink})
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=bg)
    ax = fig.add_axes([-0.06, -0.10, 1.12, 1.18], projection="3d", facecolor=bg)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((*matplotlib.colors.to_rgb("#f6e6c9"), 1.0))
        pane._axinfo["grid"].update(color="#e3d2b3", linewidth=0.8)
    ax.set_xlim(-45, 45)
    ax.set_ylim(-45, 60)
    ax.set_zlim(0, 25)
    ax.set_box_aspect((90, 105, 50))     # altitude exaggerated 2x
    ax.set_xlabel("East (m)", fontsize=14, labelpad=14)
    ax.set_ylabel("North (m)", fontsize=14, labelpad=14)
    ax.set_zlabel("Altitude (m)", fontsize=14, labelpad=8)
    ax.tick_params(labelsize=12)

    # geofence box: +-40 m horizontally, 0..alt_max vertically
    e0, e1, n0, n1, h = CFG.env.e_min, CFG.env.e_max, CFG.env.n_min, CFG.env.n_max, CFG.env.alt_max
    corners = [(e0, n0), (e1, n0), (e1, n1), (e0, n1), (e0, n0)]
    for zz in (0, h):
        ax.plot([c[0] for c in corners], [c[1] for c in corners], [zz] * 5, color=ink, ls="--", lw=1.6, alpha=0.7)
    for ce, cn in corners[:4]:
        ax.plot([ce, ce], [cn, cn], [0, h], color=ink, ls="--", lw=1.6, alpha=0.7)
    ax.text(e1, n0, h + 1.0, "geofence ±40 m, 25 m ceiling", fontsize=13, color=ink, ha="right")

    for k, (wn, we) in enumerate(M.M2_WPS, 1):
        outside = not CFG.env.inside(wn, we, M.M2_ALT)
        ax.plot([we, we], [wn, wn], [0, M.M2_ALT], color=neutral, lw=1.5)
        ax.scatter([we], [wn], [M.M2_ALT], s=260 if outside else 180, marker="X" if outside else "o",
                   color=ink if outside else neutral, edgecolor=ink, linewidth=1.5, depthshade=False)
        ax.text(we + 2, wn, M.M2_ALT + 1.5, f"P{k}" + (" (outside fence)" if outside else ""),
                fontsize=16, color=ink, fontweight="bold")
    ax.scatter([0], [0], [0], marker="^", s=160, color=ink, depthshade=False)
    ax.text(2, -3, 0.5, "home", fontsize=13, color=ink)

    trail, = ax.plot([], [], [], color=hi, lw=4.0, solid_capstyle="round")
    shadow, = ax.plot([], [], [], color=neutral, lw=2.5, alpha=0.8)
    drop, = ax.plot([], [], [], color=hi, lw=1.2, ls=":")
    drone, = ax.plot([], [], [], marker="o", ms=17, color=hi, mec=ink, mew=2.0, ls="none")
    hud = fig.text(0.035, 0.94, "", family=mono, fontsize=20, color=ink, va="top", linespacing=1.5)
    fig.text(0.965, 0.035, f"gpt-6-luna · C (tools + guard) · M2 r09 · source: {src}",
             family=mono, fontsize=12, color=ink, alpha=0.75, ha="right")

    def cmd_at(t):
        for t0, t1, txt in cmds:
            if t0 <= t <= t1:
                return txt
        return "— (waiting for agent)"

    def draw(i):
        sl = slice(0, i + 1)
        trail.set_data_3d(E[sl], N[sl], Z[sl])
        shadow.set_data_3d(E[sl], N[sl], np.zeros(i + 1))
        drop.set_data_3d([E[i], E[i]], [N[i], N[i]], [0, Z[i]])
        drone.set_data_3d([E[i]], [N[i]], [Z[i]])
        hud.set_text(f"t    {t_grid[i]:5.1f} s\nalt  {Z[i]:5.1f} m\ncmd  {cmd_at(t_grid[i])}")
        ax.view_init(elev=15, azim=-112 + 16 * t_grid[i] / T)

    peak = int(np.argmax(Z))
    draw(peak)
    fig.savefig(slides / "run09_recreation_still.png", dpi=100, facecolor=bg)
    print(f"replay: still at t={t_grid[peak]:.1f} s, alt {Z[peak]:.1f} m")
    if not video:
        plt.close(fig)
        return

    writer = FFMpegWriter(fps=FPS, codec="libx264", bitrate=-1,
                          extra_args=["-preset", "veryfast", "-pix_fmt", "yuv420p", "-crf", "18"])
    with writer.saving(fig, str(slides / "run09_recreation.mp4"), dpi=100):
        for i in range(len(t_grid)):
            draw(i)
            writer.grab_frame(facecolor=bg)
    plt.close(fig)
    print(f"replay: wrote {len(t_grid)} frames ({len(t_grid) / FPS:.1f} s)")

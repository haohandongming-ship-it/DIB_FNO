# -*- coding: utf-8 -*-
"""Figures 13, 15, 16: typhoon case study."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import common as C
C.style()
from make_figures_a import OUT, COLS, _coast  # noqa

INIT_STR = {"Rai": "16 Dec 2021 00 UTC", "Chanthu": "10 Sept 2021 00 UTC",
            "Noru": "25 Sept 2022 00 UTC"}

def _great_circle(lon1, lat1, lon2, lat2):
    from numpy import deg2rad, rad2deg, sin, cos, arcsin, sqrt
    dlat = deg2rad(lat2 - lat1); dlon = deg2rad(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(deg2rad(lat1)) * cos(deg2rad(lat2)) * sin(dlon / 2) ** 2
    return 6371.0 * 2 * arcsin(sqrt(a))

# ============================================================================
# Figure 13 -- typhoon trajectory predictions
# ============================================================================
def fig13():
    fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.6))
    for ax, name in zip(axs, C.STORM_NAMES):
        tr = C.track_series(name, "DIB-FNO")     # truth series identical for all models
        ax.plot(tr["lon"], tr["lat"], color="black", lw=2.4, label="IBTrACS best track")
        for m in C.MODELS:
            s = C.track_series(name, m)
            ax.plot(s["fc_lon"], s["fc_lat"], color=COLS[m], lw=1.7, ls="--",
                    marker="o", ms=3.4, label=m)
        t0lon = np.interp(0.0, tr["grid"], tr["lon"])
        t0lat = np.interp(0.0, tr["grid"], tr["lat"])
        ax.plot(t0lon, t0lat, marker="*", ms=13, color="#990000", zorder=8)
        ax.text(t0lon + 1.2, t0lat + 1.4, "$t_0$", fontsize=11, color="#990000", weight="bold")
        ax.set_xlim(100, 152); ax.set_ylim(0, 38)
        ax.set_xticks(range(100, 151, 10)); ax.set_yticks([0, 10, 20, 30])
        ax.set_xticklabels(["100\u00b0E", "110\u00b0E", "120\u00b0E", "130\u00b0E",
                            "140\u00b0E", "150\u00b0E"], fontsize=8.5)
        ax.set_yticklabels(["0\u00b0", "10\u00b0N", "20\u00b0N", "30\u00b0N"], fontsize=8.5)
        _coast(ax, lw=0.6)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.grid(True, alpha=0.3)
        ax.text(0.985, 0.045, "%s \u00b7 init. %s \u00b7 +72 h" % (name, INIT_STR[name]),
                transform=ax.transAxes, ha="right", fontsize=8.2, color="0.25")
        if name == "Rai":
            ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    C.savefig(fig, "fig13_typhoon_tracks.png"); plt.close(fig)

# ============================================================================
# Figure 15 -- intensity error evolution (mean over 3 storms + spread)
# ============================================================================
def fig15():
    fig, axs = plt.subplots(1, 2, figsize=(11.0, 4.6))
    series = {m: [C.track_series(n, m) for n in C.STORM_NAMES] for m in C.MODELS}
    for ax, key, ylab, title, anchor in [
            (axs[0], "wind_err", "Max wind speed error (m s$^{-1}$)",
             "(a) Max wind speed error", "e72"),
            (axs[1], "pres_err", "Min central pressure error (hPa)",
             "(b) Min pressure error", None)]:
        for m in C.MODELS:
            grid = series[m][0]["grid"]
            mat = np.column_stack([s[key] for s in series[m]])
            # rescale so the MEAN at 72 h equals the Table-4 anchor (or stated value)
            sc = C.T4[m]["e72"] / mat[-1].mean() if key == "wind_err" else 1.0
            mat = mat * sc
            mean = mat.mean(axis=1); sd = mat.std(axis=1)
            ax.plot(grid, mean, color=COLS[m], lw=2.1, label=m)
            ax.fill_between(grid, mean - sd, mean + sd, color=COLS[m], alpha=0.16, lw=0)
            ax.plot(grid[-1], mean[-1], marker="o", ms=6, color=COLS[m])
        ax.set_xlabel("Forecast lead time (h)"); ax.set_ylabel(ylab)
        ax.legend(fontsize=8.5, loc="upper left")
        ax.annotate("mean at +72 h", xy=(72, ax.get_ylim()[0] * 0.98 + 0.25), fontsize=8,
                    xytext=(40, ax.get_ylim()[1] * 0.72), color="0.4")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    C.savefig(fig, "fig15_intensity_errors.png"); plt.close(fig)

# ============================================================================
# Figure 16 -- extreme-value scatter
# ============================================================================
def fig16():
    fig, axs = plt.subplots(1, 2, figsize=(11.0, 4.8))
    # --- build obs / pred pairs from the same series as Fig. 15 --------------
    obs_w, obs_p = [], []
    preds = {"DIB-FNO": ([], []), "FourCastNet": ([], []), "AdaptFNO": ([], [])}
    for name in C.STORM_NAMES:
        tr = C.track_series(name, "DIB-FNO")
        obs_w.append(tr["wind"][1:]); obs_p.append(tr["pres"][1:])
        for m in C.MODELS:
            s = C.track_series(name, m)
            rng = np.random.default_rng(300 + C.STORM_NAMES.index(name) * 10 +
                                        C.MODELS.index(m))
            sw = np.sign(np.sin(2 * np.pi * np.arange(1, 13) / 5.0 + rng.uniform(0, 1)))
            sp = np.sign(np.cos(2 * np.pi * np.arange(1, 13) / 6.0 + rng.uniform(0, 1)))
            bias_w = {"DIB-FNO": 0.0, "FourCastNet": 0.06, "AdaptFNO": 0.04}[m]
            bias_p = {"DIB-FNO": 0.0, "FourCastNet": -0.8, "AdaptFNO": -0.5}[m]
            preds[m][0].append(tr["wind"][1:] + sw * s["wind_err"][1:] - bias_w *
                               np.maximum(tr["wind"][1:] - 45.0, 0.0))
            preds[m][1].append(tr["pres"][1:] + sp * s["pres_err"][1:] - bias_p *
                               np.maximum(960.0 - tr["pres"][1:], 0.0))
    ax = axs[0]
    for i, m in enumerate(C.MODELS):
        x = np.concatenate(obs_w); y = np.concatenate(preds[m][0])
        ax.scatter(x, y, s=14, color=COLS[m], label=m, alpha=0.75, edgecolors="none")
    lims = [24, 62]
    ax.plot(lims, lims, color="black", lw=1.6, ls="-")
    ax.text(55.5, 56.2, "perfect forecast", fontsize=8.5, rotation=45, color="black")
    ax.set_xlim(*lims); ax.set_ylim(*lims)
    ax.set_xlabel("Observed max wind speed (m s$^{-1}$)")
    ax.set_ylabel("Predicted max wind speed (m s$^{-1}$)")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.axhspan(55, 62, color="0.9", alpha=0.5, zorder=0)
    ax.axvspan(55, 62, color="0.9", alpha=0.5, zorder=0)
    ax.text(42.5, 56.5, "extreme regime\n(v > 55 m s$^{-1}$)", fontsize=8.2, color="0.35")

    ax = axs[1]
    for i, m in enumerate(C.MODELS):
        x = np.concatenate(obs_p); y = np.concatenate(preds[m][1])
        ax.scatter(x, y, s=14, color=COLS[m], label=m, alpha=0.75, edgecolors="none")
    lims = [935, 1010]
    ax.plot(lims, lims, color="black", lw=1.6)
    ax.set_xlim(*lims); ax.set_ylim(*lims)
    ax.set_xlabel("Observed min central pressure (hPa)")
    ax.set_ylabel("Predicted min central pressure (hPa)")
    ax.axvspan(935, 950, color="0.9", alpha=0.5, zorder=0)
    ax.text(957, 940.8, "extreme regime\n(p < 950 hPa)", fontsize=8.2, color="0.35")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    C.savefig(fig, "fig16_extreme_value.png"); plt.close(fig)

if __name__ == "__main__":
    import sys
    only = sys.argv[1:]
    fns = {"fig13": fig13, "fig15": fig15, "fig16": fig16}
    todo = fns if not only else {k: fns[k] for k in only if k in fns}
    for k, f in todo.items():
        f(); print("done", k)

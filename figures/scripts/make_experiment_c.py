# -*- coding: utf-8 -*-
"""Experiment C figures: the full-scale (0.25 deg) reference evaluation that
Table 2 / Table 3 of the paper report. These figures are kept strictly separate
from the Experiment P (proof-of-concept) figures; they are labeled with the
full experiment specification and must never be compared against Table 6."""
import os
import numpy as np
import matplotlib.pyplot as plt

import common as C
C.style()
from make_figures_a import OUT, COLS  # noqa

C_VARS = ["Z500", "T850", "U850", "R500"]
C_UNIT = {"Z500": r"RMSE (m$^2$ s$^{-2}$)", "T850": "RMSE (K)",
          "U850": "RMSE (m s$^{-1}$)", "R500": "Anomaly correlation"}
C_TITLE = {"Z500": "Geopotential 500 hPa", "T850": "Temperature 850 hPa",
           "U850": "Zonal wind 850 hPa", "R500": "Relative humidity 500 hPa"}


def _t3_row(model):
    return C.T3[model]


# ----------------------------------------------------------------------------
# figC1 -- Table 3 as bar charts with error bars
# ----------------------------------------------------------------------------
def figC1():
    mdl = C.MODELS
    fig, axs = plt.subplots(1, 4, figsize=(13.6, 4.0))
    spec = [("z500", "z500_s", "Z500 RMSE", "%.1f", "Z500 RMSE (m$^2$ s$^{-2}$)"),
            ("t850", "t850_s", "T850 RMSE", "%.2f", "T850 RMSE (K)"),
            ("u850", "u850_s", "U850 RMSE", "%.2f", "U850 RMSE (m s$^{-1}$)"),
            ("r500", "r500_s", "R500 ACC", "%.2f", "R500 anomaly correlation")]
    for ax, (key, key_s, name, fmt, ylab) in zip(axs, spec):
        vals = [C.T3[m][key] for m in mdl]
        errs = [C.T3[m][key_s] for m in mdl]
        span = max(vals) - min(vals)
        lo = min(vals) - 0.35 * span
        hi = max(v + e for v, e in zip(vals, errs)) + 0.30 * span
        bars = ax.bar(range(3), vals, yerr=errs, capsize=5, width=0.55,
                      color=[COLS[m] for m in mdl], edgecolor="black", lw=0.6,
                      error_kw=dict(ecolor="black", lw=1.2))
        for b, v, e in zip(bars, vals, errs):
            ax.text(b.get_x() + b.get_width() / 2, v + e + 0.16 * span, fmt % v,
                    ha="center", va="bottom", fontsize=9.5, weight="bold")
        ax.set_xticks(range(3)); ax.set_xticklabels(mdl, fontsize=9.5)
        ax.set_ylim(lo, hi)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    C.savefig(fig, "figC1_global_skill_reference.png"); plt.close(fig)


# ----------------------------------------------------------------------------
# figC2 -- lead-time curves whose 1-10-day mean matches Table 3 exactly
# ----------------------------------------------------------------------------
def figC2():
    day = np.arange(1.0, 10.01, 1.0)
    coarse = np.linspace(1, 10, 181)
    mdl = C.MODELS
    fig, axs = plt.subplots(2, 2, figsize=(12.0, 7.6))
    panels = [(axs[0, 0], "Z500", "RMSE", "Z500 RMSE (m$^2$ s$^{-2}$)"),
              (axs[0, 1], "T850", "RMSE", "T850 RMSE (K)"),
              (axs[1, 0], "U850", "RMSE", "U850 RMSE (m s$^{-1}$)"),
              (axs[1, 1], "R500", "ACC", "R500 anomaly correlation")]
    for k, (ax, v, metric, ylab) in enumerate(panels):
        fn = C.c_rmse_curve if metric == "RMSE" else C.c_acc_curve
        ymin = np.inf
        for m in mdl:
            y = fn(v, m, coarse)
            ymin = min(ymin, y.min())
            ax.plot(coarse, y, color=COLS[m], lw=2, label=m)
            ax.axhline(C.T3[m][C.T3_VAR_MAP[v]], color=COLS[m], ls=":", lw=1.0)
            for d in (1, 5, 10):
                ax.plot(d, fn(v, m, float(d)), "o", ms=4.5, color=COLS[m])
        ax.set_xlabel("Lead time (days)")
        ax.set_ylabel(ylab)
        if metric == "ACC":
            ax.set_ylim(max(0.30, ymin * 0.96), 1.02)
        if k == 0:
            ax.legend(fontsize=8.5, loc="upper left")
        ax.text(0.985, 0.965, "dotted lines = Table 3 values", transform=ax.transAxes,
                fontsize=8, ha="right", va="top", color="0.4")
    fig.tight_layout(rect=(0, 0.06, 1, 0.945))
    C.savefig(fig, "figC2_lead_time_reference.png"); plt.close(fig)


# ----------------------------------------------------------------------------
# figC3 -- Table 2 complexity at the Experiment C scale
# ----------------------------------------------------------------------------
def figC3():
    mdl = C.MODELS
    params = [0.24, 0.08, 0.08]
    flops = [0.01, 0.01, 0.01]
    rel = [1.00, 1.01, 1.00]
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.9))
    for ax, vals, title, fmt, ylab, ylim in [
            (axs[0], params, "(a) Parameters (M)", "%.2f", "Parameters (M)", None),
            (axs[1], flops, "(b) Estimated FLOPs (G)", "%.2f", "FLOPs (G)", (0.0, 0.012)),
            (axs[2], rel, "(c) Relative complexity", "%.2f\u00d7", "relative FLOPs", None)]:
        bars = ax.bar(range(3), vals, color=[COLS[m] for m in mdl], width=0.55,
                      edgecolor="black", lw=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.06, fmt % v, ha="center",
                    fontsize=10, weight="bold")
        ax.set_xticks(range(3)); ax.set_xticklabels(mdl, fontsize=9)
        if ylim:
            ax.set_ylim(*ylim)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    C.savefig(fig, "figC3_complexity_reference.png"); plt.close(fig)


if __name__ == "__main__":
    import sys
    only = sys.argv[1:]
    fns = {"figC1": figC1, "figC2": figC2, "figC3": figC3}
    todo = fns if not only else {k: fns[k] for k in only if k in fns}
    for k, f in todo.items():
        f(); print("done", k)

# -*- coding: utf-8 -*-
"""Figures 8-12, 14, 21: forecast skill and spatial maps."""
import os
import numpy as np
import matplotlib.pyplot as plt

import common as C
C.style()
from make_figures_a import OUT, COLS, _coast  # noqa

# ============================================================================
# Figure 8 -- temporal evolution RMSE / ACC per variable
# ============================================================================
def fig08():
    lead = np.arange(6, 96.01, 6.0)
    fig, axs = plt.subplots(4, 2, figsize=(11.6, 10.4))
    for r, v in enumerate(C.VARS):
        for c, (metric, fn, ylab) in enumerate([
                ("RMSE", lambda m, l: C.rmse_curve(m, v, l),
                 "%s RMSE (%s)" % (v, C.VAR_UNIT[v])),
                ("ACC", lambda m, l: C.acc_curve(m, v, l), "%s anomaly correlation" % v)]):
            ax = axs[r, c]
            for m in C.MODELS:
                y = fn(m, lead)
                ax.plot(lead, y, color=COLS[m], lw=1.9, label=m)
                for lt in (24.0, 96.0):
                    ax.plot(lt, fn(m, lt), marker="o", ms=5, color=COLS[m], zorder=5)
            ax.set_ylabel(ylab)
            ax.set_xlim(0, 96)
            if r == 3:
                ax.set_xlabel("Lead time (h)")
            if r == 0:
                ax.legend(loc="best" if c == 0 else "lower left", fontsize=8.5)
            ax.set_xticks([0, 24, 48, 72, 96])
            ax.axvline(24, color="0.6", ls=":", lw=0.8)
            ax.axvline(96, color="0.6", ls=":", lw=0.8)
    fig.tight_layout(rect=(0, 0.06, 1, 0.985))
    C.savefig(fig, "fig08_rmse_acc_lead.png"); plt.close(fig)

# ============================================================================
# Figure 9 -- autoregressive stability for Z500
# ============================================================================
def fig09():
    lead = np.arange(12, 96.01, 12.0)
    mdl = ["DIB-FNO", "FourCastNet", "AdaptFNO", "Persistence"]
    fig, axs = plt.subplots(1, 2, figsize=(11.0, 4.6))
    ax = axs[0]
    for m in mdl:
        y = C.rmse_curve(m, "Z500", lead)
        ax.plot(lead, y, color=COLS[m], lw=2, ls="--" if m == "Persistence" else "-",
                marker="o", ms=4.5, label=m)
        ax.plot([24, 96], [y[1], y[-1]], marker="o", ms=7, color=COLS[m], zorder=6)
    ax.set_xlabel("Lead time (h)"); ax.set_ylabel("Z500 RMSE (m$^2$ s$^{-2}$)")
    ax.set_xlim(0, 100); ax.set_ylim(20, 75); ax.legend(fontsize=9)
    ax.set_xticks([0, 24, 48, 72, 96])
    ax.annotate("DIB-FNO best at 24 h\n(28.9 vs 31.1 / 29.5)",
                xy=(24, 30.5), xytext=(6, 47), fontsize=8.8, color="#1f77b4",
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1))
    ax.annotate("cross-over \u2248 50-60 h", xy=(57, 55), xytext=(44, 64),
                fontsize=8.8, color="0.4",
                arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8))

    ax = axs[1]
    for m in mdl:
        y = C.acc_curve(m, "Z500", lead)
        ax.plot(lead, y, color=COLS[m], lw=2, ls="--" if m == "Persistence" else "-",
                marker="o", ms=4.5, label=m)
        ax.plot([24, 96], [y[1], y[-1]], marker="o", ms=7, color=COLS[m], zorder=6)
    ax.set_xlabel("Lead time (h)"); ax.set_ylabel("Z500 anomaly correlation")
    ax.set_xlim(0, 100); ax.set_ylim(0.90, 1.0); ax.legend(fontsize=9)
    ax.set_xticks([0, 24, 48, 72, 96])
    ax.annotate("at 96 h DIB-FNO (0.927) falls below\nFourCastNet (0.950), AdaptFNO (0.955)\n"
                "and persistence (0.958)",
                xy=(96, 0.927), xytext=(26, 0.9035), fontsize=8.8, color="0.35",
                arrowprops=dict(arrowstyle="->", color="0.35", lw=1))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    C.savefig(fig, "fig09_autoregressive_stability.png"); plt.close(fig)

# ============================================================================
# Figure 10 -- per-variable RMSE / ACC summary (24 h and 96 h)
# ============================================================================
def fig10():
    mdl = C.MODELS
    x = np.arange(4)
    w = 0.26
    fig, axs = plt.subplots(2, 2, figsize=(12.4, 7.6))
    pan = [(axs[0, 0], "RMSE", 24.0, "(a) RMSE (relative to DIB-FNO, 24 h)"),
           (axs[0, 1], "RMSE", 96.0, "(b) RMSE (relative to DIB-FNO, 96 h)"),
           (axs[1, 0], "ACC", 24.0, "(c) ACC at 24 h"),
           (axs[1, 1], "ACC", 96.0, "(d) ACC at 96 h")]
    for ax, metric, lead, title in pan:
        for k, m in enumerate(mdl):
            vals = []
            for v in C.VARS:
                s = C.model_skill(m, v)
                r = s["r24"] if lead == 24 else s["r96"]
                a = s["a24"] if lead == 24 else s["a96"]
                d = C.model_skill("DIB-FNO", v)
                dr = d["r24"] if lead == 24 else d["r96"]
                da = d["a24"] if lead == 24 else d["a96"]
                vals.append(a / da if metric == "ACC" else r / dr)
            bars = ax.bar(x + (k - 1) * w, vals, w, label=m, color=COLS[m],
                          edgecolor="black", lw=0.5)
            for b, vv in zip(bars, vals):
                if metric == "RMSE" and k > 0:
                    ax.text(b.get_x() + b.get_width() / 2, vv + 0.012, "%.3f\u00d7" % vv,
                            ha="center", fontsize=7.5)
                elif metric == "ACC":
                    ax.text(b.get_x() + b.get_width() / 2, vv + 0.0012, "%.3f" % vv,
                            ha="center", fontsize=7.5)
        ax.axhline(1.0, color="black", lw=0.8, ls=":")
        ax.set_xticks(x); ax.set_xticklabels(C.VARS)
        if metric == "RMSE":
            ax.set_ylim(0.78 if lead == 96 else 0.95, 1.13)
        else:
            ax.set_ylim(0.88, 1.005)
        ax.legend(fontsize=8.5, loc="upper left")
        if metric == "RMSE":
            ax.set_ylabel("Relative RMSE (DIB-FNO = 1), %d h lead" % int(lead))
            ax.text(0.985, 0.965, "DIB-FNO = 1.00 (reference)", transform=ax.transAxes,
                    fontsize=8.2, color="0.35", va="top", ha="right")
        else:
            ax.set_ylabel("Anomaly correlation, %d h lead" % int(lead))
    C.savefig(fig, "fig10_per_variable_skill.png"); plt.close(fig)

# ============================================================================
# map helpers
# ============================================================================
def _map_axes(fig, rect, title):
    ax = fig.add_axes(rect)
    _coast(ax)
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
    ax.set_yticks([-60, -30, 0, 30, 60])
    ax.set_xticklabels(["180\u00b0W", "120\u00b0W", "60\u00b0W", "0\u00b0", "60\u00b0E", "120\u00b0E", "180\u00b0E"],
                       fontsize=8)
    ax.set_yticklabels(["60\u00b0S", "30\u00b0S", "0\u00b0", "30\u00b0N", "60\u00b0N"], fontsize=8)
    return ax

# ============================================================================
# Figure 11 -- spatial error maps (24 h)
# ============================================================================
def fig11():
    fig = plt.figure(figsize=(11.6, 6.0))
    fields = {v: C.error_field(24.0, 100 + i) for i, v in enumerate(C.VARS)}
    for i, v in enumerate(C.VARS):
        ax = _map_axes(fig, [0.055 + (i % 2) * 0.47, 0.50 - (i // 2) * 0.37, 0.42, 0.35],
                       None)
        pc = ax.pcolormesh(C.LON2, C.LAT2, fields[v], cmap="RdBu_r", vmin=-0.4, vmax=0.4,
                           shading="nearest")
        ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
        ax.set_ylabel("%s error at +24 h (normalized)" % v, fontsize=8.2)
    fig.subplots_adjust(right=0.86)
    cax = fig.add_axes([0.88, 0.10, 0.015, 0.76])
    cb = fig.colorbar(pc, cax=cax)
    cb.set_label("Error (normalized)", fontsize=10)
    C.savefig(fig, "fig11_spatial_error.png"); plt.close(fig)

# ============================================================================
# Figure 12 -- multi-variable spatial error matrix (6 h, 24 h, 48 h)
# ============================================================================
def fig12():
    leads = [6.0, 24.0, 48.0]
    fig = plt.figure(figsize=(13.4, 8.4))
    for r, v in enumerate(C.VARS):
        for c, lt in enumerate(leads):
            ax = _map_axes(fig, [0.045 + c * 0.315, 0.70 - r * 0.19, 0.295, 0.175], None)
            pc = ax.pcolormesh(C.LON2, C.LAT2, C.error_field(lt, 200 + r * 10 + c),
                               cmap="RdBu_r", vmin=-0.4, vmax=0.4, shading="nearest")
            ax.set_xticklabels([], fontsize=8)
            ax.set_yticklabels([], fontsize=8)
            if c == 0:
                ax.set_ylabel(v, fontsize=8.5)
            if r == 0:
                ax.set_xlabel("+%d h" % int(lt), fontsize=9)
    cax = fig.add_axes([0.965, 0.09, 0.012, 0.74])
    cb = fig.colorbar(pc, cax=cax)
    cb.set_label("Error (normalized)", fontsize=10)
    C.savefig(fig, "fig12_spatial_error_matrix.png"); plt.close(fig)

# ============================================================================
# Figure 14 -- model architecture and complexity comparison (missing fig.)
# ============================================================================
def fig14():
    mdl = C.MODELS
    params = [0.313, 0.199, 0.201]
    flops = [0.316, 0.314, 0.314]
    fig, axs = plt.subplots(1, 3, figsize=(12.8, 4.6),
                            gridspec_kw=dict(width_ratios=[1.5, 1, 1]))
    # (a) backbone schematics
    ax = axs[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    rows = [(8.4, "DIB-FNO", "#fff2cc", "dynamic spectral mask + AFNO $\\times$4"),
            (5.4, "FourCastNet", "#d9e8f5", "AFNO $\\times$4, fixed low-pass truncation"),
            (2.4, "AdaptFNO", "#d5f0d5", "AFNO $\\times$4, static learnable mode weights")]
    for y, name, fc, desc in rows:
        ax.add_patch(plt.Rectangle((0.3, y - 1.2), 2.0, 1.1, fc="#9ecae1", ec="#1f5597", lw=1))
        ax.text(1.3, y - 0.65, "X", ha="center", va="center", fontsize=9)
        ax.annotate("", xy=(4.3, y - 0.65), xytext=(2.4, y - 0.65),
                    arrowprops=dict(arrowstyle="-|>", lw=1.2))
        if name == "DIB-FNO":
            ax.add_patch(plt.Rectangle((4.4, y - 1.15), 2.0, 1.0, fc=fc, ec="#b8860b", lw=1))
            ax.text(5.4, y - 0.65, "mask", ha="center", va="center", fontsize=7.5)
            ax.annotate("", xy=(6.5, y - 0.25), xytext=(5.4, y - 0.16),
                        arrowprops=dict(arrowstyle="->", lw=1.0))
        else:
            ax.add_patch(plt.Rectangle((4.4, y - 1.15), 2.0, 1.0, fc="white", ec="0.6", lw=1))
            if name == "FourCastNet":
                ax.text(5.4, y - 0.65, "fixed", ha="center", va="center", fontsize=7.5, color="0.5")
            else:
                ax.text(5.4, y - 0.65, "weights", ha="center", va="center", fontsize=7.5, color="0.5")
        ax.add_patch(plt.Rectangle((6.6, y - 1.2), 2.2, 1.1, fc="#d5f0d5", ec="#2e7d32", lw=1))
        ax.text(7.7, y - 0.65, "AFNO\n$\\times$4", ha="center", va="center", fontsize=7.5)
        ax.annotate("", xy=(9.5, y - 0.65), xytext=(8.9, y - 0.65),
                    arrowprops=dict(arrowstyle="-|>", lw=1.2))
        ax.text(0.3, y - 2.0, name + ": " + desc, fontsize=8.2, color="0.25", weight="bold")
    ax.set_ylim(0, 10.2)
    for ax, vals, title, fmt, ylab in [
            (axs[1], params, "(b) Parameters (M)", "%.3f", "Parameters (M)"),
            (axs[2], flops, "(c) Estimated FLOPs (G)", "%.3f", "FLOPs (G)")]:
        bars = ax.bar(range(3), vals, color=[COLS[m] for m in mdl], width=0.6,
                      edgecolor="black", lw=0.6)
        for b, vv in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, vv * 1.02, fmt % vv, ha="center",
                    fontsize=9.5, weight="bold")
        ax.set_xticks(range(3)); ax.set_xticklabels(mdl, fontsize=8.5)
        ax.set_ylabel(ylab)
        if title.startswith("(c)"):
            ax.set_ylim(0.30, 0.325)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    C.savefig(fig, "fig14_architecture_complexity.png"); plt.close(fig)

# ============================================================================
# Figure 21 -- single-step prediction visualization and error
# ============================================================================
def fig21():
    fields = C.make_fields()
    fig = plt.figure(figsize=(13.0, 9.6))
    titles = ["(a) Input X(t)", "(b) Prediction $\\hat{Y}$(t+6 h)",
              "(c) Truth Y(t+6 h)", "(d) Error (normalized)"]
    for r, v in enumerate(C.VARS):
        for c, key in enumerate(["input", "pred", "truth", "error"]):
            ax = _map_axes(fig, [0.035 + c * 0.243, 0.70 - r * 0.19, 0.23, 0.175], None)
            f = fields[v][key]
            if key == "error":
                pc = ax.pcolormesh(C.LON2, C.LAT2, f, cmap="RdBu_r", vmin=-0.4, vmax=0.4,
                                   shading="nearest")
            else:
                vmax = np.abs(f).max()
                pc = ax.pcolormesh(C.LON2, C.LAT2, f, cmap="RdYlBu_r",
                                   vmin=-vmax, vmax=vmax, shading="nearest")
            ax.set_xticklabels([], fontsize=8); ax.set_yticklabels([], fontsize=8)
            if c == 0:
                ax.set_ylabel(v, fontsize=8.5)
            if r == 0:
                ax.set_xlabel(titles[c], fontsize=10, labelpad=4)
    C.savefig(fig, "fig21_single_step_error.png"); plt.close(fig)

if __name__ == "__main__":
    import sys
    only = sys.argv[1:]
    fns = {"fig08": fig08, "fig09": fig09, "fig10": fig10, "fig11": fig11,
           "fig12": fig12, "fig14": fig14, "fig21": fig21}
    todo = fns if not only else {k: fns[k] for k in only if k in fns}
    for k, f in todo.items():
        f(); print("done", k)

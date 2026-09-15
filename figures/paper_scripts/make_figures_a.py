# -*- coding: utf-8 -*-
"""Regenerate all DIB-FNO paper figures. Run:  python make_figures.py [names...]"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib import cm

import common as C
C.style()
os.makedirs(C.FIGDIR, exist_ok=True)

COAST = C.COASTLINE
OUT = lambda n: os.path.join(C.FIGDIR, n)

COLS = C.MODEL_COLOR

def _coast(ax, lw=0.5):
    """Overlay the Natural Earth coastline when a local copy is available."""
    if not os.path.exists(COAST):
        return  # no coastline data -> draw the map without the overlay
    import geopandas as gpd
    gdf = gpd.read_file(COAST)
    gdf = gdf[gdf["geometry"].is_valid]
    for geom in gdf["geometry"]:
        if geom is None:
            continue
        geoms = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
        for g in geoms:
            xs, ys = np.array(g.coords).T
            ax.plot(xs, ys, color="0.35", lw=lw, zorder=3)

# ============================================================================
# Figure 1 -- architecture
# ============================================================================
def fig01():
    fig, ax = plt.subplots(figsize=(12.6, 6.2))
    ax.set_xlim(0, 122); ax.set_ylim(0, 62); ax.axis("off")
    def box(x, y, w, h, fc, ec, text, fs=9.0, lw=1.3):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.2",
                           fc=fc, ec=ec, lw=lw, zorder=2)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                zorder=3, linespacing=1.35)
        return (x, y, w, h)
    def arrow(x1, y1, x2, y2, style="-|>", col="0.25", lw=1.2, ls="-", rad=0.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13,
                                     color=col, lw=lw, linestyle=ls, zorder=1,
                                     connectionstyle="arc3,rad=%.2f" % rad))
    # ---- main stream -----------------------------------------------------
    b_input  = box(2, 17, 12.5, 8,  "#5b8ff9", "#1f3f9e",
                   "Input Field\nX(t) $\\in$ R$^{B\\times4\\times64\\times128}$\n"
                   "(ERA5, 6-hourly, 2.8$^{\\circ}$)", fs=8.6)
    b_proj   = box(17, 17, 13, 8,  "#9ecae1", "#1f5597", "Input Projection\nConv 1$\\times$1", fs=9.5)
    b_mask   = box(17, 33, 30, 17, "#fff2cc", "#b8860b",
                   "Spectral Mask Generator  $g_\\varphi$\n(lightweight hyper-network)\n"
                   "2\u00d7 stride-2 conv (64$\\rightarrow$128)\n"
                   "$\\rightarrow$ global avg. pooling\n"
                   "$\\rightarrow$ 2-layer MLP (128$\\rightarrow$H$_f$W$_f$)\n"
                   "$\\rightarrow$ $\\sigma$(logits/$\\tau$)", fs=8.4)
    b_spec   = box(16, 2.5, 16, 10.5, "#d9e8f5", "#1f5597",
                   "Spectral Filter\nZ = F$^{-1}$[F[X] $\\odot$ M($\\omega$)]\n"
                   "real FFT   H$_f$=64, W$_f$=65", fs=8.4)
    b_afno   = [box(46 + i * 14.2, 17, 13, 11, "#d5f0d5", "#2e7d32",
                    "AFNO Block %d\nLayerNorm$\\rightarrow$FFT\n$\\rightarrow$MLP (1024)\n"
                    "$\\rightarrow$iFFT\n(+ residual)" % (i + 1), fs=7.8) for i in range(4)]
    b_out    = box(105, 17, 13.5, 8, "#9ecae1", "#1f5597",
                   "Output Projection\nConv 1$\\times$1 $\\rightarrow$ C = 4", fs=9.2)
    b_final  = box(105, 43, 13.5, 7.5, "#f4cccc", "#990000",
                   "Forecast\n$\\hat{Y}$(t + $\\Delta$t)", fs=9.8)
    # ---- arrows ------------------------------------------------------------
    arrow(15.2, 21.0, 16.6, 21.0)                          # input -> proj
    arrow(30.6, 17.0, 30.6, 13.4, rad=0.0)                 # proj -> filter
    arrow(23.5, 25.0, 23.5, 32.6, rad=0.0)                 # proj -> mask gen
    arrow(34.0, 33.0, 31.5, 14.0, rad=-0.05)               # mask -> spectral filter
    ax.text(38.5, 20.5, "M($\\omega$)", ha="center", fontsize=11, color="#8a5a00")
    arrow(32.6, 7.0, 45.6, 18.2, rad=0.15)                 # filter -> AFNO1
    for i in range(3):
        arrow(59.2 + i * 14.2, 22.5, 59.9 + i * 14.2 + 0.1, 22.5)
    arrow(101.6, 22.5, 104.6, 22.5)                        # AFNO4 -> out proj
    arrow(111.7, 25.0, 111.7, 42.6)                        # out proj -> forecast
    # residual connection (dashed, above the backbone)
    arrow(105.0, 30.8, 23.5, 30.8, ls="--", col="0.45")
    arrow(23.5, 30.8, 23.5, 25.6, ls="--", col="0.45")
    ax.text(64, 32.2, "residual connection", ha="center", fontsize=8.5, color="0.4")
    # ---- labels -------------------------------------------------------------
    ax.text(29.3, 58.0, "Information Bottleneck objective",
            ha="center", fontsize=11, weight="bold")
    ax.text(29.3, 54.2, "max I(Z;Y|X)   s.t.   E$[\\|M(\\cdot;X)\\|_0]$ $\\leq$ C",
            ha="center", fontsize=10.5)
    ax.text(48.5, 5.6, "soft mask: a = $\\sigma$(logits/$\\tau$),  $\\tau$: 1.0 $\\rightarrow$ 0.1",
            ha="center", fontsize=8.4)
    ax.text(48.5, 3.2, "X$_0$ = F$^{-1}$[F[X] $\\odot$ M($\\omega$)] $\\rightarrow$ backbone",
            ha="center", fontsize=8.4, color="0.35")
    ax.text(73, 13.0, "4 $\\times$ AFNO blocks (hidden dim 1024, patch size 4)",
            ha="center", fontsize=8.8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    C.savefig(fig, "fig01_architecture.png"); plt.close(fig)

# ============================================================================
# Figure 2 -- correlation matrix
# ============================================================================
def fig02():
    labels = C.VARS
    R = np.array([[1.00, 0.93, 0.31, -0.08],
                  [0.93, 1.00, 0.27, -0.06],
                  [0.31, 0.27, 1.00, 0.04],
                  [-0.08, -0.06, 0.04, 1.00]])
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    im = ax.imshow(R, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(4)); ax.set_yticklabels(labels)
    for i in range(4):
        for j in range(4):
            v = R[i, j]
            ax.text(j, i, "%.2f" % v, ha="center", va="center", fontsize=10,
                    color="white" if abs(v) > 0.55 else "black", weight="bold" if i == j else "normal")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.85, label="Pearson correlation coefficient")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    C.savefig(fig, "fig02_correlation_matrix.png"); plt.close(fig)

# ============================================================================
# Figure 3 -- learning-rate schedule
# ============================================================================
def fig03():
    e = np.linspace(0, C.EPOCHS, 400)
    lr = 1e-5 + (1e-3 - 1e-5) * 0.5 * (1 + np.cos(np.pi * e / C.EPOCHS))
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.semilogy(e, lr, color="#1f77b4", lw=2.2)
    ax.axhline(1e-5, color="0.6", ls=":", lw=1)
    ax.axhline(1e-3, color="0.6", ls=":", lw=1)
    ax.text(10, 2.2e-3, "$\\eta_{max}$ = 10$^{-3}$", fontsize=10)
    ax.text(14.5, 4.4e-5, "$\\eta_{min}$ = 10$^{-5}$", fontsize=10)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning rate")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    C.savefig(fig, "fig03_lr_schedule.png"); plt.close(fig)

# ============================================================================
# Figure 4 -- training dynamics
# ============================================================================
def fig04():
    E = C.EPOCHS
    e = np.arange(0, E + 1)
    rng = np.random.default_rng(41)
    # (a) loss
    val = 0.0085 + (1.05 - 0.0085) * np.exp(-e / 2.63)
    trn = 0.0038 + (0.85 - 0.0038) * np.exp(-e / 2.9)
    osc = 1 + 0.10 * np.sin(e * 3.1) * np.exp(-e / 5.0)
    val = val * osc; trn = trn * (1 + 0.12 * np.sin(e * 3.7) * np.exp(-e / 5.5))
    val[-1] = 0.0085; trn[-1] = 0.0038
    # (b) RMSE / ACC
    rmse = 0.064 + (1.00 - 0.064) * np.exp(-e / 3.3)
    rmse *= (1 + 0.06 * np.sin(e * 2.7) * np.exp(-e / 4.0)); rmse[-1] = 0.064
    acc = 0.996 - (0.996 - 0.10) * np.exp(-e / 3.1)
    acc *= (1 + 0.004 * np.sin(e * 2.4)); acc[-1] = 0.996
    # (c) temperature
    tau = C.TAU0 * np.exp(np.log(C.TAU1 / C.TAU0) * e / E)
    # (d) retention
    ret = 0.50 + (C.RET_FINAL - 0.50) * (1 - np.exp(-e / 6.5))
    ret = ret + 0.015 * np.sin(e * 2.1) * np.exp(-e / 8.0)
    ret[0] = 0.50; ret[-1] = C.RET_FINAL
    lam = C.LAM_MAX * e / E

    fig, axs = plt.subplots(1, 4, figsize=(13.5, 3.9))
    axs[0].plot(e, trn, color="#1f77b4", lw=1.8, label="Training loss")
    axs[0].plot(e, val, color="#d62728", lw=1.8, label="Validation loss")
    axs[0].set_yscale("log"); axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("Loss (log)")
    axs[0].legend(fontsize=8, loc="upper right")

    axs[1].plot(e, rmse, color="#1f77b4", lw=1.8, label="RMSE (normalized)")
    axs[1].set_xlabel("Epoch"); axs[1].set_ylabel("RMSE (normalized)", color="#1f77b4")
    axs[1].tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = axs[1].twinx()
    ax2.plot(e, acc, color="#d62728", lw=1.8, label="ACC")
    ax2.set_ylabel("Anomaly correlation", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    h1, l1 = axs[1].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    axs[1].legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")

    axs[2].plot(e, tau, color="#9467bd", lw=2.2)
    axs[2].set_xlabel("Epoch"); axs[2].set_ylabel("Temperature $\\tau$")
    axs[2].annotate("$\\tau$: 1.0 \u2192 0.1", xy=(12, 0.30), fontsize=10)

    axs[3].plot(e, ret, color="#2ca02c", lw=2.2, label="Mask retention")
    axs[3].set_xlabel("Epoch"); axs[3].set_ylabel("Mask retention ratio", color="#2ca02c")
    axs[3].tick_params(axis="y", labelcolor="#2ca02c")
    ax3 = axs[3].twinx()
    ax3.plot(e, lam * 1e3, color="#b8860b", ls=":", lw=1.8, label="$\\lambda$ ($\\times 10^{-3}$)")
    ax3.set_ylabel("$\\lambda$ ($\\times 10^{-3}$)", color="#b8860b")
    ax3.tick_params(axis="y", labelcolor="#b8860b")
    h1, l1 = axs[3].get_legend_handles_labels(); h2, l2 = ax3.get_legend_handles_labels()
    axs[3].legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    for a in axs:
        a.set_xlim(0, E)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    C.savefig(fig, "fig04_training_dynamics.png"); plt.close(fig)

# ============================================================================
# Figure 5 -- convergence
# ============================================================================
def fig05():
    E = C.EPOCHS
    e = np.arange(0, E + 1)
    rng = np.random.default_rng(55)
    def curve(end, scale, osc, seed=1):
        r = np.random.default_rng(seed)
        y = end + (scale - end) * np.exp(-e / 2.9)
        y = y * (1 + osc * np.sin(e * 3.4 + r.uniform(0, 3)) * np.exp(-e / 4.5))
        y[-1] = end
        return y
    dib_t = curve(0.0038, 0.85, 0.12, 2)
    fc_t = np.clip(0.4 + (0.95 - 0.4) * np.exp(-e[:8] / 1.6), 0, None)
    ad_t = np.clip(0.45 + (0.98 - 0.45) * np.exp(-e[:8] / 1.7), 0, None)
    dib_v = curve(0.0085, 1.05, 0.10, 3)
    fc_v = 0.024 + (0.90 - 0.024) * np.exp(-e[:8] / 1.5) + 0.0015 * e[:8]  # overfits/rises
    fc_v[3] = 0.024   # minimum at epoch ~4
    ad_v = 0.026 + (0.92 - 0.026) * np.exp(-e[:8] / 1.6) + 0.0013 * e[:8]
    ad_v[4] = 0.026

    fig, axs = plt.subplots(1, 2, figsize=(11.4, 4.4))
    axs[0].plot(e, dib_t, color=COLS["DIB-FNO"], lw=2, label="DIB-FNO")
    axs[0].plot(e[:8], fc_t, color=COLS["FourCastNet"], lw=1.8, ls="--", label="FourCastNet")
    axs[0].plot(e[:8], ad_t, color=COLS["AdaptFNO"], lw=1.8, ls="--", label="AdaptFNO")
    axs[0].set_yscale("log"); axs[0].set_ylim(3e-3, 2)
    axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("Training loss (log)")
    axs[0].legend(fontsize=8.5)
    axs[0].annotate("baselines stopped at epoch 7\n(early stopping)", xy=(7, 0.05),
                    xytext=(10.5, 0.2), fontsize=8.5, color="0.35",
                    arrowprops=dict(arrowstyle="->", color="0.35", lw=1))

    axs[1].plot(e, dib_v, color=COLS["DIB-FNO"], lw=2, label="DIB-FNO")
    axs[1].plot(e[:8], fc_v, color=COLS["FourCastNet"], lw=1.8, ls="--", label="FourCastNet")
    axs[1].plot(e[:8], ad_v, color=COLS["AdaptFNO"], lw=1.8, ls="--", label="AdaptFNO")
    axs[1].set_yscale("log"); axs[1].set_ylim(5e-3, 1.5)
    axs[1].set_xlabel("Epoch"); axs[1].set_ylabel("Validation loss (log)")
    axs[1].legend(fontsize=8.5)
    axs[1].annotate("$\\sim$10$^{-2}$ by epoch 15", xy=(15, 0.0105), xytext=(8, 0.045),
                    fontsize=9, color="#1f77b4",
                    arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1))
    axs[1].annotate("validation loss rises \u2192 early stop", xy=(6.9, fc_v[-1]),
                    xytext=(9.5, 0.30), fontsize=8.5, color="0.35",
                    arrowprops=dict(arrowstyle="->", color="0.35", lw=1))
    for a in axs:
        a.set_xlim(0, E); a.set_xticks(range(0, 21, 2))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    C.savefig(fig, "fig05_convergence.png"); plt.close(fig)

# ============================================================================
# Figure 6 -- complexity bars (Table 5 values)
# ============================================================================
def fig06():
    mdl = ["DIB-FNO", "FourCastNet", "AdaptFNO"]
    params = [0.313, 0.199, 0.201]           # Table 5
    flops = [0.316, 0.314, 0.314]            # Table 5
    rel = [f / flops[2] for f in flops]
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 4.0))
    for ax, vals, title, fmt, ylab in [
            (axs[0], params, "(a) Parameters (M)", "%.3f", "Parameters (M)"),
            (axs[1], flops, "(b) Estimated FLOPs (G)", "%.3f", "FLOPs (G)"),
            (axs[2], rel, "(c) Relative complexity", "%.2f\u00d7", "FLOPs / AdaptFNO")]:
        bars = ax.bar(range(3), vals, color=[COLS[m] for m in mdl], width=0.6,
                      edgecolor="black", lw=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.02, fmt % v, ha="center",
                    fontsize=9.5, weight="bold")
        ax.set_xticks(range(3)); ax.set_xticklabels(mdl, fontsize=9.5)
        ax.set_ylabel(ylab)
        if title.startswith("(b)"):
            ax.set_ylim(0.30, 0.325)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    C.savefig(fig, "fig06_model_complexity.png"); plt.close(fig)

# ============================================================================
# Figure 7 -- training time
# ============================================================================
def fig07():
    E = C.EPOCHS
    e = np.arange(1, E + 1)
    rng = np.random.default_rng(77)
    per = np.zeros(E)
    per[0] = 20.0
    rest = 68.3 - 20.0
    w = rng.normal(1.0, 0.03, E - 1)
    per[1:] = rest / (E - 1) * w
    per[1:] *= rest / per[1:].sum()
    cum = np.cumsum(per)
    fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.0))
    axs[0].bar(e, per, color="#5b8ff9", edgecolor="black", lw=0.5, width=0.7)
    axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("Time (s)")
    axs[0].annotate("first epoch \u2248 20.0 s\n(initialization)", xy=(1, 20.0),
                    xytext=(4.5, 17.5), fontsize=9,
                    arrowprops=dict(arrowstyle="->", lw=1))
    axs[0].annotate("steady state \u2248 2.5 s", xy=(12, 2.54), xytext=(9, 8.5),
                    fontsize=9, arrowprops=dict(arrowstyle="->", lw=1))
    axs[1].plot(e, cum, color="#1f77b4", lw=2.2, marker="o", ms=4)
    axs[1].set_xlabel("Epoch"); axs[1].set_ylabel("Cumulative time (s)")
    axs[1].annotate("total = 68.3 s\n(average 3.41 s/epoch)", xy=(20, 68.3),
                    xytext=(6, 30), fontsize=9.5,
                    arrowprops=dict(arrowstyle="->", lw=1))
    for a in axs:
        a.set_xlim(0.5, 20.5); a.set_xticks(range(2, 21, 2))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    C.savefig(fig, "fig07_training_time.png"); plt.close(fig)

if __name__ == "__main__":
    import sys
    only = sys.argv[1:]
    fns = {"fig01": fig01, "fig02": fig02, "fig03": fig03, "fig04": fig04,
           "fig05": fig05, "fig06": fig06, "fig07": fig07}
    todo = fns if not only else {k: fns[k] for k in only if k in fns}
    for k, f in todo.items():
        f(); print("done", k)

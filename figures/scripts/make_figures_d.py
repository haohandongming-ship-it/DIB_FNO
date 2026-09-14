# -*- coding: utf-8 -*-
"""Figures 17-20: spectral-mask interpretation and ablation."""
import os
import numpy as np
import matplotlib.pyplot as plt

import common as C
C.style()
from make_figures_a import OUT, COLS  # noqa

MASK_CMAP = "viridis"
RNG = 1

# ============================================================================
# Figure 17 -- spectral mask interpretation
# ============================================================================
def fig17():
    a = C.soft_mask(r0=31.0, s=1.5, max_peak=0.95, target=C.RET_FINAL, seed=17)
    fig, axs = plt.subplots(1, 3, figsize=(12.4, 3.9))
    ax = axs[0]
    im = ax.imshow(a, origin="lower", aspect="auto", cmap=MASK_CMAP, vmin=0, vmax=1,
                   extent=[-32.5, 31.5, -0.5, 32.5])
    ax.set_xlabel("$k_x$ (zonal wavenumber)"); ax.set_ylabel("$k_y$ (meridional wavenumber)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="retention $a(k)$")

    ax = axs[1]
    r, p = C.radial_avg(a)
    ax.plot(r, p, color="#1f77b4", lw=2.2)
    ax.set_xlabel("Radial wavenumber $|k|$")
    ax.set_ylabel("Mean retention $\\bar{a}(|k|)$")
    ax.annotate("low-frequency core \u2248 0.95", xy=(4, 0.93), xytext=(12, 0.72),
                fontsize=8.5,
                arrowprops=dict(arrowstyle="->", lw=1))
    ax.annotate("high-wavenumber \u2192 0", xy=(38, 0.02), xytext=(27, 0.30),
                fontsize=8.5, arrowprops=dict(arrowstyle="->", lw=1))

    ax = axs[2]
    ax.hist(a.ravel(), bins=np.linspace(0, 1, 21), color="#2ca02c", edgecolor="black",
            lw=0.4, log=True)
    ax.set_xlabel("Retention $a(k)$"); ax.set_ylabel("Count (log)")
    ax.text(0.36, ax.get_ylim()[1] * 0.16, "mean = %.2f" % C.RET_FINAL, fontsize=8.5,
            color="#d62728")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    C.savefig(fig, "fig17_spectral_mask.png"); plt.close(fig)

# ============================================================================
# Figure 18 -- mask comparison across regimes
# ============================================================================
def fig18():
    a_ty = C.soft_mask(r0=33.0, s=1.8, max_peak=0.95, target=0.42, seed=18)
    a_qq = C.soft_mask(r0=27.0, s=1.8, max_peak=0.95, target=0.27, seed=19)
    diff = a_ty - a_qq
    fig, axs = plt.subplots(1, 3, figsize=(13.0, 4.0))
    for ax, arr, title, cmap, vmin, vmax in [
            (axs[0], a_ty, "(a) Typhoon regime (Rai)", MASK_CMAP, 0, 1),
            (axs[1], a_qq, "(b) Quiescent subtropical high", MASK_CMAP, 0, 1),
            (axs[2], diff, "(c) Difference  (a) \u2212 (b)", "RdBu_r", -0.3, 0.3)]:
        im = ax.imshow(arr, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                       extent=[-32.5, 31.5, -0.5, 32.5])
        ax.set_xlabel("$k_x$"); ax.set_ylabel("$k_y$")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, label="retention")
    axs[2].text(0.03, 0.97, "positive = more high-frequency\nmodes retained in typhoon regime",
                transform=axs[2].transAxes, fontsize=8.2, va="top", color="0.25")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    C.savefig(fig, "fig18_mask_regimes.png"); plt.close(fig)

# ============================================================================
# Figure 19 -- ablation over lambda
# ============================================================================
def fig19():
    lam = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
    ret = [0.88, 0.52, 0.34, 0.16, 0.07]
    rmse = [0.110, 0.113, 0.120, 0.128, 0.145]
    acc = [0.980, 0.984, 0.986, 0.983, 0.976]
    fig, axs = plt.subplots(1, 2, figsize=(10.8, 4.4))
    ax = axs[0]
    ax.scatter(ret, rmse, s=70, c=["#7f7f7f", "#9467bd", "#d62728", "#9467bd", "#7f7f7f"],
               edgecolors="black", lw=0.7, zorder=5)
    for r, m, l in zip(ret, rmse, lam):
        dx = 0.022 if l == 0.0 else 0.016
        ax.annotate("$\\lambda$=%g" % l, xy=(r, m), xytext=(r + dx, m + 0.0012),
                    fontsize=7.8, zorder=6)
    ax.set_xlim(0, 1.0); ax.set_ylim(0.104, 0.150)
    ax.set_xlabel("Mask retention ratio (fraction of modes kept)")
    ax.set_ylabel("Normalized RMSE (+6 h)")

    ax = axs[1]
    xx = np.arange(len(lam))
    bars = ax.bar(xx, rmse, width=0.6, color="#1f77b4", edgecolor="black", lw=0.6,
                  label="RMSE (normalized)")
    for b, v in zip(bars, rmse):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.0012, "%.3f" % v, ha="center",
                fontsize=8.5)
    ax.set_ylim(0.10, 0.16)
    ax.set_ylabel("Normalized RMSE (+6 h)", color="#1f77b4")
    ax.set_xticks(xx)
    ax.set_xticklabels(["0", "10$^{-4}$", "10$^{-3}$", "10$^{-2}$", "10$^{-1}$"])
    ax.set_xlabel("Sparsity weight $\\lambda$")
    ax2 = ax.twinx()
    ax2.plot(xx, acc, color="#d62728", lw=2, marker="o", ms=6, label="ACC (24 h)")
    ax2.set_ylim(0.966, 0.990)
    ax2.set_ylabel("ACC (24 h, Z500)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="lower right")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    C.savefig(fig, "fig19_sparsity_ablation.png"); plt.close(fig)

# ============================================================================
# Figure 20 -- ablation variant of the spectral mask
# ============================================================================
def fig20():
    a = C.soft_mask(r0=27.0, s=2.5, max_peak=0.30, target=0.16, seed=20)
    fig, axs = plt.subplots(1, 2, figsize=(11.0, 3.9))
    im = axs[0].imshow(a, origin="lower", aspect="auto", cmap=MASK_CMAP, vmin=0, vmax=1,
                       extent=[-32.5, 31.5, -0.5, 32.5])
    axs[0].set_xlabel("$k_x$"); axs[0].set_ylabel("$k_y$")
    fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.03, label="retention $a(k)$")
    r, p = C.radial_avg(a)
    axs[1].plot(r, p, color="#1f77b4", lw=2.2)
    axs[1].set_xlabel("Radial wavenumber $|k|$")
    axs[1].set_ylabel("Mean retention $\\bar{a}(|k|)$")
    axs[1].annotate("max \u2248 0.30 at $|k|$=0", xy=(0, 0.30), xytext=(12, 0.26),
                    fontsize=9, arrowprops=dict(arrowstyle="->", lw=1))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    C.savefig(fig, "fig20_mask_ablation.png"); plt.close(fig)

if __name__ == "__main__":
    import sys
    only = sys.argv[1:]
    fns = {"fig17": fig17, "fig18": fig18, "fig19": fig19, "fig20": fig20}
    todo = fns if not only else {k: fns[k] for k in only if k in fns}
    for k, f in todo.items():
        f(); print("done", k)

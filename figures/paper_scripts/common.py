# -*- coding: utf-8 -*-
"""
DIB-FNO paper figure regeneration -- shared data model.

Canonical protocol (single executed experiment, "proof-of-concept", PoC):
  * Input/target variables: Z500 (m^2/s^2), T850 (K), U850 (m/s), V850 (m/s)
  * Grid: 64 x 128 (lat x lon)  ->  2.8125 deg (paper wrote ~2.75 deg; see README)
  * Data: one 30-day period of 6-hourly ERA5-like states = 120 steps = 119 pairs
    80/20 chronological split -> 95 train / 24 validation
  * Trainer: AdamW lr=1e-3, cosine annealing to 1e-5, wd=1e-4, batch 16, FP32 CPU
  * Epochs: 20 (matches Figures 3-7 and Sec. 5.1 text; Sec. 4.1 "40" flagged)
  * Sparsity weight lambda: linear 0 -> 1e-3 over the first 20 epochs (warm-up)
  * Temperature tau: exponential decay 1.0 -> 0.1 over 20 epochs
  * Mask retention (validation mean, final): 0.34  (= "lambda=1e-3 -> 34%" Sec. 5.6)

All numbers taken verbatim from the paper's tables are marked with the table
reference; every other value is a deterministic reconstruction anchored to
those numbers (see PROTOCOL.md).
"""
import numpy as np
from scipy import ndimage
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Paths are resolved relative to this file, so the package stays portable:
#   <repo>/figures/paper_scripts/common.py   <- this file
#   <repo>/figures/paper/                    <- 24 paper figures (PNG + PDF)
#   <repo>/figures/paper_tables/             <- 6 re-typeset table images
#   <repo>/figures/paper_data/{tables,tracks}
# Override the repository root with the DIBFNO_FIGROOT environment variable.
# ----------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
FIGROOT = os.environ.get("DIBFNO_FIGROOT", os.path.dirname(_HERE))

FIGDIR = os.path.join(FIGROOT, "paper")
TABLEDIR = os.path.join(FIGROOT, "paper_tables")
DATADIR = os.path.join(FIGROOT, "paper_data")
TRACKDIR = os.path.join(DATADIR, "tracks")

# Natural Earth 110m coastline (public domain: https://www.naturalearthdata.com/).
# Not vendored here; point DIBFNO_COASTLINE at a local copy to enable coastlines.
# When missing, maps are drawn without the coastline overlay (no crash).
COASTLINE = os.environ.get(
    "DIBFNO_COASTLINE",
    os.path.join(FIGROOT, "paper_data", "ne_110m_coastline.geojson"),
)

# ----------------------------------------------------------------------------
# data-provenance stamps (reviewer requirement: reconstructed content must be
# visibly labeled as illustrative and must not be cited as experimental result)
# ----------------------------------------------------------------------------
STAMP = {
    "verbatim": ("DATA: verbatim from the paper (tables/text)",
                 "#1a7f37", "#e6f4ea"),
    "mixed": ("PARTLY RECONSTRUCTED: reported values + illustrative curves",
              "#8a6d00", "#fff7d6"),
    "recon": ("ILLUSTRATIVE RECONSTRUCTION \u2014 anchored to reported values; "
              "NOT model output", "#a61b00", "#fdecea"),
}

def stamp(fig, kind):
    txt, col, face = STAMP[kind]
    fig.text(0.004, 0.052, txt, color=col, fontsize=7.2, va="bottom", ha="left", zorder=50,
             bbox=dict(boxstyle="round,pad=0.25", fc=face, ec=col, lw=0.8))

def savefig(fig, name):
    """PNG + editable vector PDF, same file base name."""
    fig.savefig(os.path.join(FIGDIR, name))
    fig.savefig(os.path.join(FIGDIR, name.replace(".png", ".pdf")))

def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "mathtext.fontset": "dejavusans",
        "axes.facecolor": "white",
    })

# ----------------------------------------------------------------------------
# canonical experiment definition
# ----------------------------------------------------------------------------
VARS = ["Z500", "T850", "U850", "V850"]
VAR_UNIT = {"Z500": r"m$^2$ s$^{-2}$", "T850": "K", "U850": "m s$^{-1}$", "V850": "m s$^{-1}$"}

NLAT, NLON = 64, 128
DX = 360.0 / NLON            # 2.8125 deg
LON = np.arange(NLON) * DX - 180.0 + DX / 2.0
LAT = -90.0 + (np.arange(NLAT) + 0.5) * (180.0 / NLAT)
LON2, LAT2 = np.meshgrid(LON, LAT)

EPOCHS = 20
LAM_MAX = 1e-3
TAU0, TAU1 = 1.0, 0.1
RET_FINAL = 0.34

# ----------------------------------------------------------------------------
# skill anchors
# ----------------------------------------------------------------------------
SKILL = {
    "Z500": dict(r24=28.9,  a24=0.986, r96=67.2,  a96=0.927, sigma=173.3),
    "T850": dict(r24=0.505, a24=0.970, r96=1.174, a96=0.895, sigma=2.077),
    "U850": dict(r24=0.907, a24=0.948, r96=2.109, a96=0.852, sigma=2.860),
    "V850": dict(r24=0.907, a24=0.946, r96=2.109, a96=0.848, sigma=2.860),
}
# 24 h / 96 h values for Z500 are verbatim Table 6 (DIB-FNO row); the other
# variables are reconstructed by scaling Table 3's per-variable RMSE ratios
# onto the Table 6 anchor (see PROTOCOL.md).

MODELS = ["DIB-FNO", "FourCastNet", "AdaptFNO"]
MODEL_COLOR = {"DIB-FNO": "#1f77b4", "FourCastNet": "#ff7f0e", "AdaptFNO": "#2ca02c",
               "Persistence": "#d62728"}
T6 = {  # verbatim Table 6 (Z500)
    "DIB-FNO":     dict(r24=28.9, a24=0.986, r96=67.2, a96=0.927),
    "FourCastNet": dict(r24=31.1, a24=0.984, r96=56.8, a96=0.950),
    "AdaptFNO":    dict(r24=29.5, a24=0.986, r96=55.4, a96=0.955),
    "Persistence": dict(r24=31.7, a24=0.984, r96=52.5, a96=0.958),
}

# ----------------------------------------------------------------------------
# named experiments: two experiments, fully specified, only ONE authoritative
# ----------------------------------------------------------------------------
EXP_P = "Experiment P \u2014 proof-of-concept (authoritative main experiment)"
EXP_P_SUB = ("4 variables (Z500/T850/U850/V850) \u00b7 64\u00d7128 grid (2.8125\u00b0) \u00b7 "
             "30-day 6-hourly sample \u00b7 95/24 chronological split \u00b7 CPU \u00b7 "
             "20 epochs \u00b7 hidden 1024, patch 4 \u00b7 seed 20240517")
EXP_C = "Experiment C \u2014 full-scale reference (0.25\u00b0, 20 variables)"
EXP_C_SUB = ("ERA5 0.25\u00b0 6-hourly, 1979\u20132022 \u00b7 train 1979\u20132017, "
             "val 2018\u20132019, test 2020\u20132022 \u00b7 4\u00d7A100 \u00b7 80 epochs \u00b7 "
             "batch 32 FP16 \u00b7 hidden 1024, patch 4 \u00b7 1\u201310-day leads")
# short tags embedded in every figure (reviewer requirement: every figure must
# state the experiment resolution / variables / period / rollout / width / seed)
EXP_P_TAG = ("Experiment P \u2014 proof-of-concept (authoritative main experiment): "
             "4 variables \u00b7 64\u00d7128 (2.8125\u00b0) \u00b7 30-day 6-hourly sample \u00b7 "
             "95/24 chronological split \u00b7 CPU \u00b7 20 epochs \u00b7 hidden 1024, patch 4 \u00b7 "
             "seed 20240517")
EXP_P_TAG_SHORT = ("Experiment P (authoritative main experiment)\n"
                   "4 variables \u00b7 64\u00d7128 (2.8125\u00b0) \u00b7 30-day 6-hourly sample \u00b7 "
                   "95/24 chronological split\n"
                   "CPU \u00b7 20 epochs \u00b7 hidden 1024, patch 4 \u00b7 seed 20240517")
EXP_C_TAG = ("Experiment C \u2014 full-scale reference (NOT the main experiment): "
             "ERA5 0.25\u00b0 \u00b7 20 variables \u00b7 1979\u20132022 \u00b7 train 1979\u20132017 / "
             "val 2018\u20132019 / test 2020\u20132022 \u00b7 4\u00d7A100 \u00b7 80 epochs \u00b7 "
             "batch 32 FP16 \u00b7 hidden 1024, patch 4")
EXP_C_TAG_SHORT = ("Experiment C (reference \u2014 NOT the main experiment)\n"
                   "ERA5 0.25\u00b0 \u00b7 20 variables \u00b7 1979\u20132022 \u00b7 "
                   "train 1979\u20132017 / val 2018\u20132019 / test 2020\u20132022\n"
                   "4\u00d7A100 \u00b7 80 epochs \u00b7 batch 32 FP16 \u00b7 hidden 1024, patch 4")

# ----------------------------------------------------------------------------
# Experiment C anchors (verbatim Table 3; mean over 1-10 day leads)
# ----------------------------------------------------------------------------
T3 = {  # Z500 RMSE (m2/s2), T850 RMSE (K), U850 RMSE (m/s), R500 ACC
    "FourCastNet": dict(z500=112.4, z500_s=1.2, t850=1.82, t850_s=0.03,
                        u850=3.45, u850_s=0.05, r500=0.78, r500_s=0.01),
    "AdaptFNO":    dict(z500=108.7, z500_s=1.0, t850=1.79, t850_s=0.02,
                        u850=3.38, u850_s=0.04, r500=0.80, r500_s=0.01),
    "DIB-FNO":     dict(z500=104.2, z500_s=0.9, t850=1.74, t850_s=0.02,
                        u850=3.27, u850_s=0.04, r500=0.83, r500_s=0.01),
}
T3_VAR_MAP = {"Z500": "z500", "T850": "t850", "U850": "u850", "R500": "r500"}

def c_rmse_curve(var, model, day):
    """Experiment-C RMSE(d) = a * d^q with mean over days 1..10 == Table 3 value."""
    dv = T3_VAR_MAP[var]
    s = T3[model][dv]
    qt = {"Z500": 0.66, "T850": 0.70, "U850": 0.68, "R500": 0.68}[var]
    dd = np.arange(1.0, 10.0 + 1e-9, 1.0)
    a = s * len(dd) / np.power(dd, qt).sum()
    return a * np.power(np.asarray(day, float), qt)

def c_acc_curve(var, model, day):
    """Experiment-C ACC(d) = exp(-(d/tau)^p); tau solved so that the
    mean over days 1..10 equals the Table 3 ACC value. Table 3 reports an
    ACC column only for R500, so the same ACC curve applies to all variables."""
    target = T3[model]["r500"]
    p = 1.15
    dd = np.arange(1.0, 10.0 + 1e-9, 1.0)
    def mean_of(tau):
        return np.exp(-np.power(dd / tau, p)).mean()
    lo, hi = 1e-6, 1e6
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        if mean_of(mid) < target:
            lo = mid
        else:
            hi = mid
    tau = np.sqrt(lo * hi)
    return np.exp(-np.power(np.asarray(day, float) / tau, p))

def model_skill(model, var):
    """(r24, a24, r96, a96) for any model x variable, anchored on the tables."""
    if var == "Z500":
        return dict(T6[model])
    s = SKILL[var]
    m = T6[model]
    d = T6["DIB-FNO"]
    return dict(r24=s["r24"] * m["r24"] / d["r24"],
                a24=s["a24"] * m["a24"] / d["a24"],
                r96=s["r96"] * m["r96"] / d["r96"],
                a96=s["a96"] * m["a96"] / d["a96"])

def _q(r24, r96):
    return np.log(r96 / r24) / np.log(4.0)

def _acc_params(a24, a96):
    p = np.log(np.log(1 / a96) / np.log(1 / a24)) / np.log(4.0)
    tau = 24.0 / np.power(np.log(1.0 / a24), 1.0 / p)
    return p, tau

def rmse_curve(model, var, lead):
    s = model_skill(model, var)
    return s["r24"] * np.power(np.asarray(lead, float) / 24.0, _q(s["r24"], s["r96"]))

def acc_curve(model, var, lead):
    s = model_skill(model, var)
    p, tau = _acc_params(s["a24"], s["a96"])
    return np.exp(-np.power(np.asarray(lead, float) / tau, p))

# ----------------------------------------------------------------------------
# masks (real-FFT frequency plane 64 x 65)
# ----------------------------------------------------------------------------
HF, WF = NLAT, NLON // 2 + 1
KX = np.arange(-NLAT // 2, NLAT // 2)
KY = np.arange(0, WF)
KXX, KYY = np.meshgrid(KX, KY)
RAD = np.sqrt(KXX ** 2 + KYY ** 2)
RP_MAX = float(RAD.max())

def _logistic(r0, s):
    return 1.0 / (1.0 + np.exp((RAD - r0) / s))

def soft_mask(r0=31.0, s=1.5, max_peak=0.95, target=None, seed=1, sharp=True):
    """isotropic low-pass mask on the real-FFT plane (64 x 65):
    a(k) = max_peak * sigmoid((r0 - |k|)/s), optionnally renormalized so that
    mean(a) == target (equivalent to the mask-retention ratio)."""
    rng = np.random.default_rng(seed)
    prof = max_peak * _logistic(r0, max(s, 1.0))
    if target is not None:
        c = target / max(prof.mean(), 1e-9)
        prof = np.clip(prof * c, 0, 1)
        if prof.mean() < target:      # clip lost mass -> sparse speckle
            need = target - prof.mean()
            far = prof < 0.1
            n_far = int(far.sum())
            if n_far:
                amp = 0.10 * rng.random(n_far)
                cov = min(1.0, need * prof.size / max(amp.mean() * n_far, 1e-9))
                sel = rng.random(n_far) < cov
                idx = np.flatnonzero(far)[sel]
                prof.ravel()[idx] = np.maximum(prof.ravel()[idx], amp[sel])
    return np.clip(prof, 0, 1)

def radial_avg(a):
    rr, aa = RAD.ravel(), a.ravel()
    bins = np.arange(0, RP_MAX + 1.0, 1.0)
    cents, prof = [], []
    for i in range(len(bins) - 1):
        m = (rr >= bins[i]) & (rr < bins[i + 1])
        if m.sum() > 0:
            cents.append(0.5 * (bins[i] + bins[i + 1]))
            prof.append(aa[m].mean())
    return np.array(cents), np.array(prof)

# ----------------------------------------------------------------------------
# meteorological fields (synthetic reconstructions, deterministic)
# ----------------------------------------------------------------------------
def _smooth_noise(rng, sigma_px=3.0):
    n = rng.normal(size=(NLAT, NLON))
    n = ndimage.gaussian_filter(n, sigma_px, mode="wrap")
    return n / (n.std() + 1e-12)

def wave_field(rng, waves_lon=(2, 4, 7), waves_lat=(1, 2, 3), amp=(1.0, 0.55, 0.3),
               mean_lat=0.0, grad_lat=0.0, texture=0.015):
    f = np.zeros((NLAT, NLON))
    latn = (LAT2 + 90.0) / 180.0
    f += mean_lat + grad_lat * (0.5 - latn)
    phases = rng.uniform(0, 2 * np.pi, 12)
    for nw, wl, am, i in zip(waves_lon, waves_lat, amp, range(len(waves_lon))):
        f += am * np.sin(2 * np.pi * nw * (LON2 + 180.0) / 360.0 + phases[i]) \
             * np.cos(np.pi * wl * (latn - 0.5) + phases[i + 3])
    f += texture * _smooth_noise(rng, 2.0)
    return f

def make_fields(seed=7):
    """dict var -> (input, truth, prediction, error) all normalized to unit std"""
    rng = np.random.default_rng(seed)
    out = {}
    for v in VARS:
        if v == "Z500":
            t = wave_field(rng, mean_lat=0.0, grad_lat=-2.0)
        elif v == "T850":
            t = wave_field(rng, mean_lat=0.0, grad_lat=-3.2, waves_lon=(2, 3, 6),
                           waves_lat=(1, 2, 2), amp=(1.0, 0.5, 0.25))
        elif v == "U850":
            jet = np.exp(-0.5 * ((LAT2 - 45.0) / 12.0) ** 2) \
                - 0.35 * np.exp(-0.5 * ((LAT2 + 40.0) / 12.0) ** 2)
            t = 1.4 * jet + wave_field(rng, waves_lon=(3, 5), waves_lat=(2, 4),
                                       amp=(0.35, 0.2), texture=0.01)
        else:
            t = wave_field(rng, waves_lon=(3, 6), waves_lat=(2, 3),
                           amp=(0.5, 0.3), grad_lat=0.8)
        t = (t - t.mean()) / t.std()
        # fine-scale texture = the high-frequency content that a Fourier
        # low-pass model cannot reproduce (over-smoothing, Sec. 5.7)
        fine = 0.12 * _smooth_noise(rng, 0.8)
        truth = t + fine
        inp = (t + 0.08 * _smooth_noise(rng, 1.5))
        inp = (inp - inp.mean()) / inp.std()
        pred = t                                               # smoothed field
        err = np.clip(fine, -0.4, 0.4)
        out[v] = dict(input=inp, truth=truth, pred=pred, error=err)
    return out

def error_field(lead_h, seed, amp=0.4):
    rng = np.random.default_rng(seed + int(lead_h))
    scale = 0.10 + 0.045 * (lead_h / 48.0)
    e = _smooth_noise(rng, 1.2) * scale
    return np.clip(e, -amp, amp)

# ----------------------------------------------------------------------------
# typhoon best tracks (JMA, via Digital Typhoon) + synthetic forecasts
# ----------------------------------------------------------------------------
STORM_NAMES = ["Rai", "Chanthu", "Noru"]
STORM_INIT = {"Rai": "2021-12-16 00:00:00", "Chanthu": "2021-09-10 00:00:00",
              "Noru": "2022-09-25 00:00:00"}
T4 = {  # verbatim Table 4
    "FourCastNet": dict(e24=85.0, e48=210.0, e72=7.2),
    "AdaptFNO":    dict(e24=78.0, e48=192.0, e72=6.8),
    "DIB-FNO":     dict(e24=62.0, e48=158.0, e72=5.4),
}

def load_best_tracks():
    import pandas as pd
    tracks = {}
    for name in STORM_NAMES:
        df = pd.read_csv(os.path.join(TRACKDIR, "%s_besttrack.csv" % name.lower()))
        t0 = pd.to_datetime(df["time_utc"])
        tracks[name] = dict(time=t0.values,
                            lon=df["lon"].values, lat=df["lat"].values,
                            wind=df["wind_ms"].values, pres=df["pres_hpa"].values)
    return tracks

def track_series(name, model, seed=11):
    """best-track truth + model forecast on a 0..72 h grid (6 h steps),
    initialized at STORM_INIT. Returns dict with grid, best, fc_lon/lat,
    wind_err, pres_err (m/s, hPa)."""
    import pandas as pd
    tr = load_best_tracks()[name]
    init = pd.Timestamp(STORM_INIT[name])
    t = tr["time"]
    h = (t - np.datetime64(init)) / np.timedelta64(1, "h")
    grid = np.arange(0, 72.01, 6.0)
    lon_t = np.interp(grid, h, tr["lon"], left=np.nan, right=np.nan)
    lat_t = np.interp(grid, h, tr["lat"], left=np.nan, right=np.nan)
    wind_t = np.interp(grid, h, tr["wind"], left=np.nan, right=np.nan)
    pres_t = np.interp(grid, h, tr["pres"], left=np.nan, right=np.nan)

    rng = np.random.default_rng(seed + STORM_NAMES.index(name) * 100)
    m = T4[model]
    # ---- displacement -------------------------------------------------------
    # mean great-circle error at lead t equals dist_km(t); the Table-4 anchors
    # (e24, e48) are therefore reproduced exactly at 24 h and 48 h.
    k = np.log(m["e48"] / m["e24"]) / np.log(2.0)          # growth exponent
    dist_km = m["e24"] * np.power(np.maximum(grid, 1) / 24.0, k)
    # decompose into along-/cross-track components: angle phi per (storm, model)
    phi = {"DIB-FNO": 0.20, "FourCastNet": 0.95, "AdaptFNO": 0.70}[model] * \
          rng.uniform(-1.0, 1.0)                            # fixed per storm
    wig = 0.25 * np.sin(2 * np.pi * grid / 40.0 + rng.uniform(0, 2 * np.pi)) \
        + 0.15 * ndimage.gaussian_filter1d(rng.normal(0, 1, grid.size), 1.3)
    th = phi + wig
    dlat = np.gradient(np.where(np.isnan(lat_t), 0, lat_t))
    dlon = np.gradient(np.where(np.isnan(lon_t), 0, lon_t))
    heading = np.arctan2(dlat, dlon)                        # storm motion direction
    lat_scale = np.cos(np.radians(np.nanmean(lat_t))) + 1e-9
    dl = dist_km * np.cos(th) / 111.0                       # deg lat, along motion
    dc = dist_km * np.sin(th) / (111.0 * lat_scale)         # deg lon, cross motion
    fc_lat = lat_t + dl * np.sin(heading) + dc * np.cos(heading)
    fc_lon = lon_t + dl * np.cos(heading) - dc * np.sin(heading)
    # ---- intensity errors: zero at t=0, equal to the Table-4 anchor at 72 h
    osc = {"DIB-FNO": 0.10, "FourCastNet": 0.18, "AdaptFNO": 0.16}[model]
    pows = {"DIB-FNO": 1.80, "FourCastNet": 1.35, "AdaptFNO": 1.40}[model]
    wind_err = m["e72"] * np.power(grid / 72.0, pows) \
        * (1.0 + osc * np.sin(2 * np.pi * grid / 26.0 + rng.uniform(0, 2 * np.pi)))
    p_anchor = {"DIB-FNO": 1.6, "FourCastNet": 3.0, "AdaptFNO": 2.6}[model]
    pres_err = p_anchor * np.power(grid / 72.0, pows) \
        * (1.0 + 0.6 * osc * np.sin(2 * np.pi * grid / 30.0 + rng.uniform(0, 2 * np.pi)))
    return dict(grid=grid, lon=lon_t, lat=lat_t, wind=wind_t, pres=pres_t,
                fc_lon=fc_lon, fc_lat=fc_lat, wind_err=wind_err, pres_err=pres_err)

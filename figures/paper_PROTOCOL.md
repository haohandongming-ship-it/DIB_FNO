# DIB-FNO Figure-Regeneration Protocol

This document specifies the experiment protocol used to regenerate **all figures** of
the paper. Per the review requirement, the results of the paper are organized into
**two explicitly named experiments**; each is fully specified (resolution, variable
count, training years, test years, rollout mode, model width, random seed) and every
figure carries the name of the experiment it belongs to. **Only Experiment P is the
authoritative main experiment** that may support the abstract/conclusion claims.
Experiment C reproduces the paper's Table 2 / Table 3 numbers and is kept strictly
separate; the two must never be compared or averaged.

---

## 1. Named experiments

### Experiment P — proof-of-concept (AUTHORITATIVE MAIN EXPERIMENT)
All figures `fig01`–`fig21` belong to this experiment.

| Field | Specification |
|---|---|
| Name / role | Experiment P — proof-of-concept (authoritative main experiment) |
| Input variables | Z500 (m² s⁻²), T850 (K), U850 (m s⁻¹), V850 (m s⁻¹) — C = 4 |
| Resolution | 64 × 128 (lat × lon) = 2.8125° grid |
| Training data | 30-day period, 6-hourly ERA5-like states: 120 steps → 119 consecutive pairs; 80/20 chronological split → 95 train / 24 validation |
| Training / test years | one 30-day sample of the 1979–2022 record (the paper does not name the specific period; flagged as a gap) |
| Rollout mode | autoregressive, 6-hourly steps, up to 96 h (4 steps = 24 h, 16 steps = 96 h); typhoon case: analysis-initiated 72 h rollouts |
| Model | 4×AFNO backbone, hidden dim 1024, patch size 4 + mask generator (2 stride-2 convs 64→128, global avg-pool, 2-layer MLP 128→64×65, σ(logits/τ)) |
| Model width | hidden 1024; mask generator channels 64→128 |
| Hardware / precision | CPU, FP32 (the paper's 4×A100 + FP16 description belongs to Experiment C) |
| Optimizer | AdamW, lr 1e-3, cosine → 1e-5, wd 1e-4, batch 16 |
| Epochs | 20 |
| λ schedule | linear 0 → 1e-3 over epochs 1–20 (warm-up) |
| τ schedule | exponential 1.0 → 0.1 over 20 epochs |
| Random seed | 20240517 master seed (fixes every stochastic draw in the regenerated material; the original run's seed is not reported by the paper) |
| Anchors (verbatim) | Table 5 (complexity), Table 6 (24 h / 96 h skill), Table 4 (typhoon), §5.1 (68.3 s), §5.6 (λ = 1e-3 → 34%) |

### Experiment C — full-scale reference (0.25°, 20 variables)
Reproduces the paper's **Table 2 / Table 3** numbers (figures `figC1`–`figC3`).

| Field | Specification |
|---|---|
| Name / role | Experiment C — full-scale reference (NOT the main experiment; reported separately so that Table 3 does not silently contradict Table 6) |
| Input variables | 20 prognostic variables of FourCastNet (§4.4; evaluation reported for Z500, T850, U850, R500 in Table 3) |
| Resolution | ERA5 0.25° 6-hourly (≈720 × 1440 grid) |
| Training years | 1979–2017 |
| Validation years | 2018–2019 |
| Test years | 2020–2022 |
| Rollout mode | 6-hourly autoregressive rollouts evaluated at 1–10-day leads; Table-3 metrics = latitude-weighted mean over the 1–10-day window |
| Model | same DIB-FNO design (4×AFNO, hidden 1024, patch 4) described in §4.4 |
| Hardware / precision | 4×A100, mixed precision FP16, batch 32 |
| Epochs / time | 80 epochs, ≈2 days |
| Random seed | 20240517 (regenerated material only; original seed not reported) |
| Anchors (verbatim) | Table 2 (complexity), Table 3 (skill) |

**Authoritative statement.** Experiment P is the ONLY experiment that may be cited in
the abstract and conclusion as the result of this work. The Experiment C figures are
included because the paper reports Table 3; they are labeled "reference" on every
figure, and their numbers (104.2 m²/s², 1.74 K, 3.27 m s⁻¹, ACC 0.83) must not be used
in any "consistently outperforms" claim until a C-scale run is actually performed.

## 2. Figure → experiment mapping

| Figures | Experiment | Verbatim anchors | Content |
|---|---|---|---|
| fig01–fig03 | P | §4.1 | architecture, correlation matrix (Z500–T850 = 0.93), LR schedule |
| fig04–fig07 | P | §5.1 (68.3 s), Table 5 | training dynamics, convergence, complexity, timing |
| fig08–fig10 | P | Table 6 | RMSE/ACC vs lead, per-variable skill (mixed result incl. the 96 h regression) |
| fig11, fig12, fig21 | P | §5.2 (±0.4 error bound) | spatial error maps |
| fig13, fig15, fig16 | P | Table 4, JMA best tracks | typhoon case study |
| fig14 | P | Table 5 | missing figure regenerated (architecture + complexity) |
| fig17–fig20 | P | §5.6 (34%, 0.30 peak, ±0.3 diff) | mask interpretability + ablation |
| figC1–figC3 | C | Table 3, Table 2 | bar summary, lead-time curves (1–10-day means = Table 3 exactly), complexity |

## 3. Experiment P data and curve models

### Skill anchors (verbatim Table 6, Z500)

| Model | 24 h RMSE / ACC | 96 h RMSE / ACC |
|---|---|---|
| DIB-FNO | 28.9 / 0.986 | 67.2 / 0.927 |
| FourCastNet | 31.1 / 0.984 | 56.8 / 0.950 |
| AdaptFNO | 29.5 / 0.986 | 55.4 / 0.955 |
| Persistence | 31.7 / 0.984 | 52.5 / 0.958 |

Per-variable anchors (Z500 from Table 6; T850/U850 by applying Table 3's per-variable
RMSE ratios to the Table 6 anchor; σ implied by RMSE = σ·√(1 − ACC²)):

| Variable | 24 h RMSE | 24 h ACC | 96 h RMSE | 96 h ACC | σ (implied) |
|---|---|---|---|---|---|
| Z500 | 28.9 m² s⁻² | 0.986 | 67.2 m² s⁻² | 0.927 | 173.3 m² s⁻² |
| T850 | 0.505 K | 0.970 | 1.174 K | 0.895 | 2.077 K |
| U850 | 0.907 m s⁻¹ | 0.948 | 2.109 m s⁻¹ | 0.852 | 2.860 m s⁻¹ |
| V850 | 0.907 m s⁻¹ | 0.946 | 2.109 m s⁻¹ | 0.848 | 2.860 m s⁻¹ |

Baseline per-variable skill = DIB-FNO per-variable values × the model-level Table-6
ratios (e.g., FourCastNet 96 h RMSE ratio = 56.8/67.2 = 0.845).

### Curve models (anchored exactly at the table endpoints)

* RMSE(lead):  R(t) = r₂₄ · (t/24)^q,  q = ln(r₉₆/r₂₄)/ln(4).
* ACC(lead):   A(t) = exp(−(t/τ)^p),  p = ln(ln(1/a₉₆)/ln(1/a₂₄))/ln(4),  τ = 24·ln(1/a₂₄)^(−1/p).
* Validation loss: val(e) = 0.0085 + 1.0415·exp(−e/2.63) (+0.10 decaying sine) ⇒ val(15) ≈ 0.012 ("near 10⁻² by epoch 15"), val(20) = 0.0085.
* Training loss: trn(e) = 0.0038 + 0.8462·exp(−e/2.9).
* Normalized +6 h RMSE (fig04b): 1.00 → 0.064; ACC: 0.10 → 0.996.
* Training time (fig07): 20.0 s (epoch 1) + 19 × 2.54 s = 68.3 s total (3.41 s/epoch mean).
* Baselines (fig05): minima at epochs 4–5, early-stop at epoch 7 (validation loss rising).
* Ablation (fig19): λ ∈ {0, 10⁻⁴, 10⁻³, 10⁻², 10⁻¹} ⇒ retention {0.88, 0.52, 0.34, 0.16, 0.07}; normalized RMSE {0.110, 0.113, 0.120, 0.128, 0.145}; ACC {0.980, 0.984, 0.986, 0.983, 0.976} (λ = 0 → 0.11 min RMSE; λ = 10⁻³ → 0.12 with peak ACC — verbatim §5.6).

## 4. Masks (fig17, fig18, fig20)

Masks live on the 64 × 65 real-FFT plane; a(k) = max_peak·σ((r₀ − |k|)/s), then
renormalized so mean(a) equals the retention ratio (mean(a) ≡ "fraction of modes
retained", §5.6):

| Figure | r₀ | s | max_peak | mean(a) | meaning |
|---|---|---|---|---|---|
| 17 (validation mean) | 31 | 1.5 | 0.95 | **0.34** | λ = 1e-3 optimal run |
| 18(a) typhoon (Rai) | 33 | 1.8 | 0.95 | 0.42 | more high-wavenumber modes |
| 18(b) quiescent high | 27 | 1.8 | 0.95 | 0.27 | nearer low-pass |
| 18(c) difference | — | — | — | range ±0.3 | §5.5 verbatim amplitude |
| 20 (λ = 1e-2 variant) | 27 | 2.5 | 0.30 | 0.16 | radial average peaks at 0.30 (§5.6) |

Note on geometry: on a 64×65 real-FFT plane, 34% retention ⇔ isotropic half-disk of
radius ≈ 31 grid units; the small "yellow blob" of the original fig17(a) is consistent
with ≈4% retention only, so the regenerated masks are geometrically consistent with the
stated retention (34% ⇔ r₀ ≈ 31).

## 5. Typhoon evaluation (fig13, fig15, fig16)

* Best tracks: JMA 6-hourly best track of IBTrACS v04r01 (WMO/TOKYO series), from the
  Digital Typhoon archive (`data/tracks/*_besttrack.csv`). Wind = 10-min max (kt) × 0.5144 ⇒ m s⁻¹.
* Init times (storm near peak intensity): Rai 2021-12-16 00Z; Chanthu 2021-09-10 00Z; Noru 2022-09-25 00Z.
* Rollout: 72 h at 6-h steps for all typhoon figures (the paper's 230 h axis is removed as inconsistent with the stated 72 h rollout).
* Track errors (verbatim Table 4): DIB-FNO 62 ± 9 km (24 h), 158 ± 18 km (48 h); FourCastNet 85 ± 12 / 210 ± 25; AdaptFNO 78 ± 10 / 192 ± 22.
* Synthetic forecast placements: displacement(t) = e(t)·[cosθ·u + sinθ·v], e(t) = e₂₄·(t/24)^k, k = ln(e₄₈/e₂₄)/ln(2) ⇒ mean great-circle error equals Table 4 at 24 h and 48 h; θ per (storm, model): DIB ≤ 0.20 rad, FourCastNet ≤ 0.95, AdaptFNO ≤ 0.70 rad + smooth wiggle.
* Intensity errors (fig15): |err|(t) = a·(t/72)^p·(1 + osc·sin(2πt/26 + φ)), p = 1.80 / 1.35 / 1.40, osc = 0.10 / 0.18 / 0.16 (DIB / FourCastNet / AdaptFNO), renormalized so that the 3-storm mean at 72 h equals Table 4 (5.4 / 7.2 / 6.8 m s⁻¹). Pressure errors: 1.6 / 3.0 / 2.6 hPa at 72 h (fig15 narrative).
* fig16: predicted vs observed (max wind, min pressure) from the same per-storm error series; baselines include a mild under-prediction tendency at the extremes.

## 6. Spatial error maps (fig11, fig12, fig21)

* 64×128 physical lon/lat grid (centers −178.59° … 178.59°, −87.19° … 87.19°), Natural Earth 110m coastline, no projection (as in the original paper).
* Error fields: smooth Gaussian-correlated noise, σ_norm ≈ 0.10–0.15 growing with lead, clipped so |error| ≤ 0.4 (verbatim §5.2); common [−0.4, 0.4] RdBu_r scale.
* fig21 fields: synthetic planetary-wave fields (zonal mean + harmonics 2–7, meridional envelopes; U850 with westerly jets at 45°N and ~40°S), unit-std normalized; truth = base + fine texture (σ = 0.12); prediction = base (smoothed); error = truth − prediction ⇒ over-smoothing narrative (§5.7).

## 7. Experiment C figures (figC1–figC3)

* figC1: Table 3 as grouped bar charts with the reported ± std error bars
  (Z500 RMSE 104.2 ± 0.9 etc.).
* figC2: lead-time curves constrained so that the 1–10-day mean equals Table 3 exactly:
  RMSE(d) = a·d^q with a = value·10/Σd^q (q = 0.66, 0.70, 0.68 for Z500, T850, U850);
  ACC(d) = exp(−(d/τ)^1.15) with τ solved by bisection so that the 1–10-day mean equals
  the Table-3 ACC value. Table 3 reports ACC only for R500, hence a single ACC curve
  family (0.78 / 0.80 / 0.83) is used.
* figC3: Table 2 complexity (0.24M / 0.08M / 0.08M; 0.01G FLOPs; relative 1.00× / 1.01× / 1.00×).

## 8. Values: verbatim vs reconstructed

**Verbatim (kept exactly as printed in the paper):** Table 1 rows; Tables 2–6 all
numbers; §5.1 68.3 s / 20.0 s / 3.41 s per-epoch; "validation loss near 10⁻² by epoch 15";
§5.2 "errors bounded by −0.4 and 0.4"; §5.3 SSIM 0.72 vs 0.58, 10°×10° window;
§5.5 "difference map fluctuations between −0.3 and 0.3"; §5.6 "λ = 10⁻³ → 34% retained",
"λ = 0 → RMSE 0.11", "λ = 10⁻³ → RMSE ≈ 0.12, peak ACC", "radial average maximum
retention probability of only 0.3".

**Reconstructed, anchored to the verbatim numbers** (deterministic seeds; documented
above): all epoch/lead-time curves; per-variable P-scale skill derived from Tables 3+6
ratios; four-variable correlation matrix (fig02; Z500–T850 = 0.93 from Table 1, others
estimated from typical ERA5 field relations); mask fields (radii from retention ratios);
synthetic forecast placements/intensities; error maps; Experiment C lead-time curves
(means pinned to Table 3).

**External data used:** JMA best tracks (Rai 202112, Chanthu 202114, Noru 202216) via
Digital Typhoon; Natural Earth 110m coastline.

## 8.1 Per-figure data-provenance classification (reviewer requirement)

Every figure carries a printed stamp (bottom-left, English) declaring its provenance;
the stamp is one of:

* `DATA: verbatim from the paper (tables/text)` — figure presents only reported values.
* `PARTLY RECONSTRUCTED: reported values + illustrative curves (see README / PROTOCOL)`
* `ILLUSTRATIVE RECONSTRUCTION — synthesized to match reported values; NOT model output,
  do not cite as a result`

| Figure | Stamp | Reported (verbatim) part | Reconstructed part |
|---|---|---|---|
| fig01 | verbatim | architecture described in §4.1 | — (schematic) |
| fig02 | mixed | Z500–T850 = +0.93 (Table 1) | other off-diagonal correlations (typical ERA5 estimates, printed in-figure) |
| fig03 | mixed | η max/min, cosine schedule, 20 epochs | — (schedule computed from stated formula) |
| fig04 | recon | — | loss, RMSE/ACC, τ, retention curves (anchors: §5.1, §5.6) |
| fig05 | recon | "validation near 10⁻² by epoch 15" | all curves; early-stop epoch from §5.1 text |
| fig06 | verbatim | Table 5 | — |
| fig07 | mixed | 68.3 s total, 20.0 s first epoch, 3.41 s mean | per-epoch split |
| fig08 | recon | Table-6 anchors | curves |
| fig09 | recon | Table-6 anchors | curves |
| fig10 | recon | Table-6 anchors | curves |
| fig11 | recon | ±0.4 error bound (§5.2) | fields |
| fig12 | recon | ±0.4 error bound (§5.2) | fields |
| fig13 | mixed | JMA best tracks (real data), Table-4 errors | forecast lines |
| fig14 | verbatim | Table 5 | — (schematic) |
| fig15 | recon | Table-4 72-h anchors | error series |
| fig16 | recon | Table-4 anchors | scatter points |
| fig17 | recon | 34% retention (§5.6) | mask fields |
| fig18 | recon | 0.42/0.27, ±0.3 (§5.5) | mask fields |
| fig19 | mixed | λ=0 → 0.11; λ=10⁻³ → 0.12, 34%, peak ACC (§5.6) | intermediate λ points |
| fig20 | recon | radial max 0.30 (§5.6) | mask |
| fig21 | recon | ±0.4 error bound (§5.7) | fields |
| figC1 | verbatim | Table 3 | — |
| figC2 | recon | Table-3 means | curves |
| figC3 | verbatim | Table 2 | — |
| tables 01–06 | verbatim | Tables 1–6 | — |

**Usage rule:** figures stamped `recon` are **illustrative only** — they are not
experimental results, must appear (if at all) as labeled supplementary material, and the
corresponding over-claiming text (e.g., "consistently outperforms", "effectively resolves
error-accumulation collapse") must be removed from the main text. Figures stamped
`verbatim` may be cited in the main text as reported values; `mixed` figures may be cited
only for their reported components.

## 9. Editable delivery (per PDF-processing guidelines)

* Each figure is delivered twice: **300 dpi PNG** (`figures/paper/*.png`) and **vector PDF**
  (`figures/paper/*.pdf`, same base name, fully editable in any PDF editor).
* All anchor values are plain CSVs (`figures/paper_data/tables/table01..06.csv`,
  `figures/paper_data/tracks/*.csv`); every figure is produced by an executable script
  (`figures/paper_scripts/`) with a fixed seed, so any figure can be re-generated or
  re-styled without re-drawing by hand.
* Fonts: DejaVu Sans + mathtext (Latin/Greek glyphs only; no CJK glyphs inside figures,
  so transcription to PDF/LaTeX pipelines will not produce mojibake).

## 10. Reproducibility

All random processes use fixed `default_rng` seeds (scripts/common.py + each figure);
figure order does not affect results. Re-run: `make_figures_a.py` → `make_figures_b.py`
→ `make_figures_c.py` → `make_figures_d.py` → `make_experiment_c.py` → `make_tables.py`.
Python ≥ 3.10, numpy/scipy/matplotlib/pandas/geopandas. All outputs: 300 dpi PNG +
vector PDF with English-only labels (DejaVu Sans + mathtext).

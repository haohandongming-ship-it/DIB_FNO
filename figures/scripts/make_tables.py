# -*- coding: utf-8 -*-
"""Regenerate paper tables as CSV data files (+ clean table images)."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import common as C
C.style()

DATA = os.path.join(C.DATADIR, "tables")
FIGT = os.path.join(C.FIGDIR, "tables")
os.makedirs(DATA, exist_ok=True)
os.makedirs(FIGT, exist_ok=True)

TABLES = {}

# ---- Table 1 (verbatim) ----------------------------------------------------
TABLES["table01_variable_correlations"] = pd.DataFrame([
    ["Z500 & T850", "+0.93"],
    ["Temperature across levels", "+0.94 to +0.99"],
], columns=["Variable pair", "Correlation"])

# ---- Table 2 (verbatim, production-scale estimates) ------------------------
TABLES["table02_model_complexity_production"] = pd.DataFrame([
    ["DIB-FNO", 0.24, 0.01, "1.00x"],
    ["FourCastNet", 0.08, 0.01, "1.01x"],
    ["AdaptFNO", 0.08, 0.01, "1.00x"],
], columns=["Model", "Params (M)", "Estimated FLOPs (G)", "Relative"])

# ---- Table 3 (verbatim, global forecast metrics) ---------------------------
TABLES["table03_global_forecast_skill"] = pd.DataFrame([
    ["FourCastNet", "112.4 ± 1.2", "1.82 ± 0.03", "3.45 ± 0.05", "0.78 ± 0.01"],
    ["AdaptFNO", "108.7 ± 1.0", "1.79 ± 0.02", "3.38 ± 0.04", "0.80 ± 0.01"],
    ["DIB-FNO", "104.2 ± 0.9", "1.74 ± 0.02", "3.27 ± 0.04", "0.83 ± 0.01"],
], columns=["Model", "Z500 RMSE (m²/s²)", "T850 RMSE (K)", "U850 RMSE (m/s)", "R500 ACC"])

# ---- Table 4 (verbatim, typhoon errors) ------------------------------------
TABLES["table04_typhoon_forecast_errors"] = pd.DataFrame([
    ["FourCastNet", "85 ± 12", "210 ± 25", "7.2 ± 1.1"],
    ["AdaptFNO", "78 ± 10", "192 ± 22", "6.8 ± 1.0"],
    ["DIB-FNO", "62 ± 9", "158 ± 18", "5.4 ± 0.8"],
], columns=["Model", "24-h track error (km)", "48-h track error (km)", "72-h intensity error (m/s)"])

# ---- Table 5 (verbatim, PoC complexity) ------------------------------------
TABLES["table05_model_complexity_poc"] = pd.DataFrame([
    ["DIB-FNO", 0.313, 0.316],
    ["FourCastNet", 0.199, 0.314],
    ["AdaptFNO", 0.201, 0.314],
], columns=["Model", "Params (M)", "FLOPs (G)"])

# ---- Table 6 (verbatim, multi-step skill) ----------------------------------
TABLES["table06_multistep_skill"] = pd.DataFrame([
    ["DIB-FNO", "28.9 / 0.986", "67.2 / 0.927"],
    ["FourCastNet", "31.1 / 0.984", "56.8 / 0.950"],
    ["AdaptFNO", "29.5 / 0.986", "55.4 / 0.955"],
    ["Persistence", "31.7 / 0.984", "52.5 / 0.958"],
], columns=["Model", "24h RMSE/ACC", "96h RMSE/ACC"])

for name, df in TABLES.items():
    df.to_csv(os.path.join(DATA, name + ".csv"), index=False)

# ---- render as clean tables ------------------------------------------------
def render(df, title, fname, colw=None):
    fig, ax = plt.subplots(figsize=(6.2, 0.42 * (len(df) + 2)))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center",
                   cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.45)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("0.6")
        if r == 0:
            cell.set_facecolor("#dbe7f5"); cell.set_text_props(weight="bold")
    ax.set_title(title, fontsize=10.5, pad=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGT, fname), bbox_inches="tight")
    plt.close(fig)

render(TABLES["table01_variable_correlations"], "Table 1: Variable correlations",
       "table01.png")
render(TABLES["table02_model_complexity_production"],
       "Table 2: Model complexity (production-scale estimate)", "table02.png")
render(TABLES["table03_global_forecast_skill"],
       "Table 3: Global forecast metrics (average over 1-10 day lead times)",
       "table03.png")
render(TABLES["table04_typhoon_forecast_errors"],
       "Table 4: Typhoon forecast errors", "table04.png")
render(TABLES["table05_model_complexity_poc"],
       "Table 5: Model complexity (proof-of-concept run)", "table05.png")
render(TABLES["table06_multistep_skill"],
       "Table 6: Multi-step skill (RMSE / ACC)", "table06.png")
print("tables written to", DATA, "and", FIGT)

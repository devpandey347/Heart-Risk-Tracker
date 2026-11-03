"""
eda.py — Exploratory Data Analysis for UCI Cleveland Heart Disease Dataset
==========================================================================
Run this script from the project root:
    python ml/src/eda.py

Outputs:
    ml/reports/eda_summary.txt   — Text summary of all findings
    ml/reports/figures/          — All visualisation PNG files

This script NEVER modifies the original dataset file.
It is purely observational; no model is trained here.
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
# pyrefly: ignore[missing-import]
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────────────
# 0.  PATHS
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent          # ml/src/
ML_DIR       = SCRIPT_DIR.parent                        # ml/
PROJECT_ROOT = ML_DIR.parent                            # project root

RAW_DATA     = ML_DIR / "data" / "raw" / "processed.cleveland.data"
REPORT_DIR   = ML_DIR / "reports"
FIGURES_DIR  = REPORT_DIR / "figures"
REPORT_TXT   = REPORT_DIR / "eda_summary.txt"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 0b. GUARD: make sure the dataset exists
# ─────────────────────────────────────────────────────────────────────────────
if not RAW_DATA.exists():
    print(f"\n❌  Dataset not found at: {RAW_DATA}")
    print("    Please place 'processed.cleveland.data' in ml/data/raw/ and re-run.")
    sys.exit(1)

print(f"✅  Dataset located at: {RAW_DATA}")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  COLUMN NAMES  (per UCI documentation)
#     https://archive.ics.uci.edu/dataset/45/heart+disease
# ─────────────────────────────────────────────────────────────────────────────
COLUMN_NAMES = [
    "age",       # Age in years                               (numerical)
    "sex",       # 1 = male, 0 = female                       (categorical)
    "cp",        # Chest pain type (1–4)                      (categorical)
    "trestbps",  # Resting blood pressure (mmHg)              (numerical)
    "chol",      # Serum cholesterol (mg/dl)                  (numerical)
    "fbs",       # Fasting blood sugar > 120 mg/dl (1=true)   (categorical)
    "restecg",   # Resting ECG results (0–2)                  (categorical)
    "thalach",   # Maximum heart rate achieved                 (numerical)
    "exang",     # Exercise-induced angina (1=yes)             (categorical)
    "oldpeak",   # ST depression induced by exercise           (numerical)
    "slope",     # Slope of peak exercise ST segment (1–3)    (categorical)
    "ca",        # Number of major vessels coloured (0–3)     (numerical*)
    "thal",      # Thalassemia (3=normal, 6=fixed, 7=reversable) (categorical*)
    "target",    # Diagnosis — 0 = no disease, 1–4 = disease  (target)
]

# Features that are truly categorical in meaning
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
NUMERICAL_FEATURES   = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]

# ─────────────────────────────────────────────────────────────────────────────
# 2.  LOAD — treat '?' as NaN immediately
# ─────────────────────────────────────────────────────────────────────────────
df_raw = pd.read_csv(
    RAW_DATA,
    header=None,
    names=COLUMN_NAMES,
    na_values=["?"],
)

print(f"\n📐  Raw shape: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  DATASET SUMMARY (printed + written to file)
# ─────────────────────────────────────────────────────────────────────────────
lines = []   # accumulate all text for the report file

def h(title):
    """Print and store a section header."""
    sep = "=" * 70
    block = f"\n{sep}\n  {title}\n{sep}"
    print(block)
    lines.append(block)

def p(text=""):
    """Print and store a line of text."""
    print(text)
    lines.append(str(text))

# ── 3a. Basic info ──────────────────────────────────────────────────────────
h("SECTION 1 — DATASET BASICS")
p(f"  File        : {RAW_DATA}")
p(f"  Delimiter   : comma (',')")
p(f"  Rows        : {df_raw.shape[0]}")
p(f"  Columns     : {df_raw.shape[1]}  (13 features + 1 target)")
p(f"  Missing marker in file : '?'  → loaded as NaN")

# ── 3b. Data types ──────────────────────────────────────────────────────────
h("SECTION 2 — DATA TYPES")
dtype_str = df_raw.dtypes.to_string()
p(dtype_str)
p()
p("  NOTE: All columns load as float64 because '?' was the only non-numeric value.")
p("        Categorical features are stored as floats by convention in this dataset.")

# ── 3c. Missing values ──────────────────────────────────────────────────────
h("SECTION 3 — MISSING VALUES")
missing = df_raw.isnull().sum()
missing_pct = (missing / len(df_raw) * 100).round(2)
missing_df = pd.DataFrame({"count": missing, "percent": missing_pct})
p(missing_df[missing_df["count"] > 0].to_string())
p()
p(f"  Total missing cells : {missing.sum()}")
p(f"  Rows with ≥1 missing: {df_raw.isnull().any(axis=1).sum()}")
p()
p("  IMPORTANT: 'ca' (vessels coloured by fluoroscopy) and 'thal' (thalassemia)")
p("  have missing values.  These will be imputed in the preprocessing pipeline")
p("  (Chunk 3).  DO NOT drop these rows during EDA — we observe only.")

# ── 3d. Descriptive statistics ───────────────────────────────────────────────
h("SECTION 4 — DESCRIPTIVE STATISTICS (numerical features)")
desc = df_raw[NUMERICAL_FEATURES].describe().round(3)
p(desc.to_string())

# ── 3e. Original target distribution ────────────────────────────────────────
h("SECTION 5 — ORIGINAL TARGET DISTRIBUTION (values 0–4)")
orig_counts = df_raw["target"].value_counts().sort_index()
for val, cnt in orig_counts.items():
    p(f"  target = {int(val)} → {cnt:3d} rows  ({cnt/len(df_raw)*100:.1f}%)")

# ── 3f. Binary target mapping ───────────────────────────────────────────────
h("SECTION 6 — BINARY TARGET MAPPING")
p("  Per UCI documentation and standard ML practice:")
p("  • target == 0  → No heart disease (absence)")
p("  • target >= 1  → Heart disease present (values 1, 2, 3, 4 all → 1)")
p()
p("  This is the standard Cleveland subset convention used in virtually")
p("  all published benchmarks on this dataset.")
p()

df = df_raw.copy()
df["target_binary"] = (df["target"] >= 1).astype(int)

binary_counts = df["target_binary"].value_counts().sort_index()
for val, cnt in binary_counts.items():
    label = "No disease" if val == 0 else "Disease present"
    p(f"  Binary target = {val} ({label}): {cnt:3d} rows  ({cnt/len(df)*100:.1f}%)")

imbalance_ratio = binary_counts[1] / binary_counts[0]
p()
p(f"  Class ratio (disease:no-disease) = {imbalance_ratio:.2f} : 1")
p("  → Dataset is mildly imbalanced; will use stratified splits in training.")

# ── 3g. Categorical value ranges ────────────────────────────────────────────
h("SECTION 7 — CATEGORICAL FEATURE VALUE RANGES")
for col in CATEGORICAL_FEATURES:
    vals = sorted(df_raw[col].dropna().unique())
    p(f"  {col:<10}: {vals}")

# ── 3h. Feature-by-feature description ──────────────────────────────────────
h("SECTION 8 — FEATURE DESCRIPTIONS (UCI)")
descriptions = {
    "age"      : "Age in years (29–77 in this dataset)",
    "sex"      : "1 = male, 0 = female",
    "cp"       : "Chest pain: 1=typical angina, 2=atypical, 3=non-anginal, 4=asymptomatic",
    "trestbps" : "Resting blood pressure in mmHg on admission to hospital",
    "chol"     : "Serum cholesterol in mg/dl",
    "fbs"      : "Fasting blood sugar > 120 mg/dl: 1=true, 0=false",
    "restecg"  : "Resting ECG: 0=normal, 1=ST-T wave abnormality, 2=left ventricular hypertrophy",
    "thalach"  : "Maximum heart rate achieved (bpm)",
    "exang"    : "Exercise-induced angina: 1=yes, 0=no",
    "oldpeak"  : "ST depression induced by exercise relative to rest",
    "slope"    : "Slope of peak exercise ST segment: 1=up, 2=flat, 3=down",
    "ca"       : "Number of major vessels (0–3) coloured by fluoroscopy  ⚠ has missing values",
    "thal"     : "Thalassemia: 3=normal, 6=fixed defect, 7=reversible defect  ⚠ has missing values",
    "target"   : "Diagnosis (0=no disease; 1–4=disease; mapped to binary for ML)",
}
for feat, desc_text in descriptions.items():
    p(f"  {feat:<10}: {desc_text}")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  STYLE SETUP FOR PLOTS
# ─────────────────────────────────────────────────────────────────────────────
PALETTE_BINARY = ["#2DD4BF", "#F97316"]   # teal = no disease, orange = disease
PALETTE_CAT    = sns.color_palette("muted", 10)
TEXT_COLOR     = "#1E293B"
GRID_COLOR     = "#E2E8F0"

plt.rcParams.update({
    "figure.facecolor"  : "white",
    "axes.facecolor"    : "#F8FAFC",
    "axes.edgecolor"    : "#CBD5E1",
    "axes.labelcolor"   : TEXT_COLOR,
    "axes.titlesize"    : 12,
    "axes.titleweight"  : "bold",
    "axes.labelsize"    : 10,
    "xtick.color"       : TEXT_COLOR,
    "ytick.color"       : TEXT_COLOR,
    "xtick.labelsize"   : 9,
    "ytick.labelsize"   : 9,
    "legend.fontsize"   : 9,
    "figure.titlesize"  : 14,
    "figure.titleweight": "bold",
    "grid.color"        : GRID_COLOR,
    "grid.linewidth"    : 0.8,
    "font.family"       : "sans-serif",
})

def save_fig(name, fig=None):
    """Save the current (or given) figure and close it."""
    path = FIGURES_DIR / name
    (fig or plt).savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  💾  Saved → {path.name}")
    return path.name

saved_figures = []   # track all saved filenames

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — TARGET CLASS DISTRIBUTION (binary)
# ─────────────────────────────────────────────────────────────────────────────
h("FIGURE 1 — Target class distribution (binary)")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle("Heart Disease — Target Class Distribution", y=1.02)

# Left: bar chart
labels = ["No Disease (0)", "Disease (1)"]
counts = [binary_counts[0], binary_counts[1]]
bars = axes[0].bar(labels, counts, color=PALETTE_BINARY, width=0.5, edgecolor="white", linewidth=1.5)
for bar, cnt in zip(bars, counts):
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 3, str(cnt),
                 ha="center", va="bottom", fontweight="bold", fontsize=11)
axes[0].set_ylabel("Number of Patients")
axes[0].set_title("Count by Class")
axes[0].set_ylim(0, max(counts) * 1.15)
axes[0].grid(axis="y", linestyle="--")
axes[0].set_axisbelow(True)

# Right: pie chart with original breakdown
orig_labels = [f"Original target={int(v)}" for v in orig_counts.index]
orig_sizes  = orig_counts.values
pie_colors  = ["#2DD4BF"] + ["#F97316", "#FB923C", "#FDBA74", "#FED7AA"][: len(orig_sizes) - 1]
wedges, texts, autotexts = axes[1].pie(
    orig_sizes, labels=orig_labels, colors=pie_colors,
    autopct="%1.1f%%", startangle=140,
    textprops={"fontsize": 8},
    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
)
for at in autotexts:
    at.set_fontsize(8)
axes[1].set_title("Original Target Values (0–4)")

plt.tight_layout()
saved_figures.append(save_fig("fig1_target_distribution.png"))

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — MISSING VALUES
# ─────────────────────────────────────────────────────────────────────────────
h("FIGURE 2 — Missing values by feature")

missing_plot = missing[missing > 0]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.barh(missing_plot.index, missing_plot.values, color="#F97316", edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, missing_plot.values):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
            f"{val} rows ({val/len(df_raw)*100:.1f}%)",
            va="center", fontsize=10, fontweight="bold")
ax.set_xlabel("Number of Missing Values")
ax.set_title("Features with Missing Values ('?' in original file)")
ax.set_xlim(0, missing_plot.max() * 1.5)
ax.grid(axis="x", linestyle="--")
ax.set_axisbelow(True)

# Annotate all-good features
ax.text(0.98, 0.05,
        f"11 of 13 features: 0 missing values\n2 features have '?' (ca, thal)",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, color="#64748B",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F1F5F9", edgecolor="#CBD5E1"))

plt.tight_layout()
saved_figures.append(save_fig("fig2_missing_values.png"))

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — NUMERICAL FEATURE DISTRIBUTIONS (histograms + KDE)
# ─────────────────────────────────────────────────────────────────────────────
h("FIGURE 3 — Numerical feature distributions")

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Numerical Feature Distributions", y=1.01)
axes = axes.flatten()

for i, col in enumerate(NUMERICAL_FEATURES):
    ax = axes[i]
    data = df_raw[col].dropna()
    ax.hist(data, bins=25, color="#2DD4BF", edgecolor="white",
            linewidth=0.8, density=True, alpha=0.8, label="Histogram")
    data.plot.kde(ax=ax, color="#0F172A", linewidth=2, label="KDE")
    ax.axvline(data.mean(),   color="#F97316", linewidth=1.5, linestyle="--", label=f"Mean={data.mean():.1f}")
    ax.axvline(data.median(), color="#7C3AED", linewidth=1.5, linestyle=":",  label=f"Median={data.median():.1f}")
    ax.set_title(col)
    ax.set_xlabel(col)
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

plt.tight_layout()
saved_figures.append(save_fig("fig3_numerical_distributions.png"))

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — NUMERICAL FEATURES GROUPED BY BINARY TARGET (box plots)
# ─────────────────────────────────────────────────────────────────────────────
h("FIGURE 4 — Numerical features by target class (boxplots)")

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Numerical Features — Distribution by Heart Disease Status", y=1.01)
axes = axes.flatten()

for i, col in enumerate(NUMERICAL_FEATURES):
    ax = axes[i]
    groups = [
        df[df["target_binary"] == 0][col].dropna(),
        df[df["target_binary"] == 1][col].dropna(),
    ]
    bp = ax.boxplot(groups, patch_artist=True, widths=0.5,
                    medianprops=dict(color="#0F172A", linewidth=2))
    for patch, color in zip(bp["boxes"], PALETTE_BINARY):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_xticklabels(["No Disease", "Disease"])
    ax.set_title(col)
    ax.set_ylabel(col)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

plt.tight_layout()
saved_figures.append(save_fig("fig4_numerical_by_target.png"))

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — CATEGORICAL FEATURE DISTRIBUTIONS BY TARGET (stacked bar)
# ─────────────────────────────────────────────────────────────────────────────
h("FIGURE 5 — Categorical feature distributions by target class")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Categorical Features — Proportion by Heart Disease Status", y=1.01)
axes = axes.flatten()

for i, col in enumerate(CATEGORICAL_FEATURES):
    ax = axes[i]
    ct = df.groupby([col, "target_binary"]).size().unstack(fill_value=0)
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    ct_pct.plot(
        kind="bar", stacked=True, ax=ax,
        color=PALETTE_BINARY, edgecolor="white", linewidth=0.8, width=0.65,
    )
    ax.set_title(col)
    ax.set_xlabel("")
    ax.set_ylabel("% of patients")
    ax.set_ylim(0, 115)
    ax.legend(["No Disease", "Disease"], fontsize=7, loc="upper right")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

# Hide the unused 8th subplot
if len(CATEGORICAL_FEATURES) < len(axes):
    for j in range(len(CATEGORICAL_FEATURES), len(axes)):
        axes[j].set_visible(False)

plt.tight_layout()
saved_figures.append(save_fig("fig5_categorical_by_target.png"))

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6 — CORRELATION HEATMAP (Pearson — numerical features only)
# ─────────────────────────────────────────────────────────────────────────────
h("FIGURE 6 — Correlation heatmap")

# Use only numerical features + binary target for correlation
corr_cols  = NUMERICAL_FEATURES + ["target_binary"]
corr_df    = df[corr_cols].copy()
corr_matrix = corr_df.corr(method="pearson")

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)   # show lower triangle only
sns.heatmap(
    corr_matrix, mask=mask,
    annot=True, fmt=".2f", linewidths=0.5, linecolor="#E2E8F0",
    cmap="RdBu_r", center=0, vmin=-1, vmax=1,
    ax=ax, annot_kws={"size": 9},
    cbar_kws={"shrink": 0.8},
)
ax.set_title(
    "Pearson Correlation Heatmap — Numerical Features + Binary Target\n"
    "⚠ Pearson is computed on raw values; categorical features excluded",
    fontsize=11,
)
ax.tick_params(axis="both", labelsize=9)
plt.tight_layout()
saved_figures.append(save_fig("fig6_correlation_heatmap.png"))

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 7 — NUMERICAL FEATURE PAIRPLOT (coloured by binary target)
# ─────────────────────────────────────────────────────────────────────────────
h("FIGURE 7 — Pairplot (numerical features, coloured by target)")

pair_cols = ["age", "thalach", "oldpeak", "chol", "target_binary"]
pair_df   = df[pair_cols].dropna().copy()
pair_df["Heart Disease"] = pair_df["target_binary"].map({0: "No Disease", 1: "Disease"})

pg = sns.pairplot(
    pair_df, hue="Heart Disease",
    vars=["age", "thalach", "oldpeak", "chol"],
    palette={"No Disease": "#2DD4BF", "Disease": "#F97316"},
    plot_kws={"alpha": 0.55, "s": 20},
    diag_kind="kde",
    corner=True,
)
pg.figure.suptitle("Pairplot — Key Numerical Features by Target Class", y=1.01, fontsize=13, fontweight="bold")
pg.figure.set_facecolor("white")
saved_figures.append(save_fig("fig7_pairplot.png", pg.figure))

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 8 — AGE DISTRIBUTION BY TARGET (KDE overlay)
# ─────────────────────────────────────────────────────────────────────────────
h("FIGURE 8 — Age distribution by target class (KDE)")

fig, ax = plt.subplots(figsize=(9, 5))
for val, label, color in zip([0, 1], ["No Disease", "Disease"], PALETTE_BINARY):
    subset = df[df["target_binary"] == val]["age"].dropna()
    subset.plot.kde(ax=ax, color=color, linewidth=2.5, label=f"{label} (n={len(subset)})")
    ax.axvline(subset.mean(), color=color, linestyle="--", linewidth=1.2,
               label=f"Mean age = {subset.mean():.1f} yrs")

ax.set_title("Age Distribution by Heart Disease Status")
ax.set_xlabel("Age (years)")
ax.set_ylabel("Density")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.6)
ax.set_axisbelow(True)
plt.tight_layout()
saved_figures.append(save_fig("fig8_age_by_target.png"))

# ─────────────────────────────────────────────────────────────────────────────
# 5.  FINAL TEXT REPORT SECTION
# ─────────────────────────────────────────────────────────────────────────────
h("SECTION 9 — KEY OBSERVATIONS (EDA Summary)")
p("  1. DATASET SIZE: 303 patients, 13 features + 1 target — a small but well-studied dataset.")
p("  2. MISSING VALUES: Only 'ca' and 'thal' have missing values (6 rows total — ~2% of data).")
p("     These will be imputed in the preprocessing pipeline, not dropped.")
p("  3. CLASS BALANCE: 164 no-disease (54.1%) vs 139 disease (45.9%) — mildly imbalanced.")
p("     Standard stratified train/test split will handle this adequately.")
p("  4. NUMERICAL FEATURES:")
p("     • 'thalach' (max heart rate) tends to be LOWER in disease patients — a strong signal.")
p("     • 'oldpeak' (ST depression) tends to be HIGHER in disease patients — another strong signal.")
p("     • 'age' and 'trestbps' show moderate separation between classes.")
p("     • 'chol' shows minimal separation — a weaker discriminator here.")
p("  5. CATEGORICAL FEATURES:")
p("     • 'cp' (chest pain type 4 — asymptomatic) is strongly associated with disease.")
p("     • 'exang' (exercise angina = 1) is highly associated with disease.")
p("     • 'sex' (male=1) shows higher disease prevalence in this dataset.")
p("     • 'thal' (reversible defect = 7) is strongly associated with disease.")
p("  6. CORRELATION CAVEAT: Pearson correlation is computed only for numerical features.")
p("     Categorical features (like 'sex', 'cp', 'thal') require Cramér's V or phi-coefficient")
p("     for meaningful correlation estimates — not done here to keep the script simple.")
p("  7. DATASET LIMITATIONS:")
p("     • Only 303 patients — generalization to the real world is limited.")
p("     • All patients are from one hospital (Cleveland Clinic Foundation, 1988).")
p("     • Gender, ethnicity, and age distributions may not represent the general population.")
p("     • This dataset CANNOT be used for clinical diagnosis — educational use only.")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  SAVE TEXT REPORT
# ─────────────────────────────────────────────────────────────────────────────
report_text = "\n".join(lines)
with open(REPORT_TXT, "w", encoding="utf-8") as f:
    f.write(report_text)

print(f"\n📄  Text report saved → {REPORT_TXT}")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  FINAL CONSOLE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  EDA COMPLETE — All outputs written successfully")
print("=" * 70)
print(f"\n  Rows         : {df_raw.shape[0]}")
print(f"  Columns      : {df_raw.shape[1]}")
print(f"  Missing cells: {missing.sum()} (in 'ca' and 'thal' only)")
print(f"  Class 0      : {binary_counts[0]} patients ({binary_counts[0]/len(df)*100:.1f}%) — no disease")
print(f"  Class 1      : {binary_counts[1]} patients ({binary_counts[1]/len(df)*100:.1f}%) — disease present")
print(f"\n  Figures saved ({len(saved_figures)}):")
for name in saved_figures:
    print(f"    • ml/reports/figures/{name}")
print(f"\n  Text report  : ml/reports/eda_summary.txt")
print()
print("  ➡  Next step: Run Chunk 3 — ML model training (after approval).")
print("=" * 70)

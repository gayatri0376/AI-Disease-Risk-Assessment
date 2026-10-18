"""
analyze_data.py — Phase 3: Exploratory Data Analysis
=====================================================
AI-Based Anemia Screening and Blood Report Interpretation System
College Project — Educational Use Only

PURPOSE
-------
This script performs a thorough analysis of the anemia_types.csv dataset.
It does NOT modify the original CSV. It only reads data, prints findings,
and saves plots and a text report to ml_model/reports/.

HOW TO RUN (from project root):
    source .venv/bin/activate
    python ml_model/src/analyze_data.py

OUTPUTS
-------
- Console: readable summary of all findings
- ml_model/reports/diagnosis_distribution.png  — bar chart of class counts
- ml_model/reports/feature_boxplots.png        — boxplots for all 14 features
- ml_model/reports/analysis_report.txt         — text copy of the full report
"""

# ===========================================================================
# SECTION 0: Imports and Path Setup
# ===========================================================================

# Use 'Agg' backend so matplotlib works on macOS without a GUI display window
import matplotlib
matplotlib.use("Agg")

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------------------------
# Project-root-safe path setup.
# This script is at: ml_model/src/analyze_data.py
# So the project root is two levels up from this file.
# Using __file__ makes this work regardless of which directory you run from.
# ---------------------------------------------------------------------------
THIS_FILE    = Path(__file__).resolve()
SRC_DIR      = THIS_FILE.parent          # ml_model/src/
ML_DIR       = SRC_DIR.parent            # ml_model/
PROJECT_ROOT = ML_DIR.parent             # AI-Anemia-Screening/

DATASET_PATH = ML_DIR / "dataset" / "anemia_types.csv"
REPORTS_DIR  = ML_DIR / "reports"

# Create reports directory if it does not exist yet
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# We will collect all printed text so we can also save it to a file
_report_lines = []


def report(text=""):
    """Print a line to the console and store it for the saved text report."""
    print(text)
    _report_lines.append(str(text))


# ===========================================================================
# SECTION 1: Load the Dataset
# ===========================================================================

def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load the CSV from the given path.
    Raises a clear, helpful error if the file is missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Dataset not found at:\n  {path}\n"
            f"Please run this script from the project root:\n"
            f"    python ml_model/src/analyze_data.py\n"
        )
    df = pd.read_csv(path)
    return df


# ===========================================================================
# SECTION 2: Basic Dataset Information
# ===========================================================================

def print_basic_info(df: pd.DataFrame):
    """Print shape, column names, data types, and sample rows."""
    report("=" * 70)
    report("SECTION 1: BASIC DATASET INFORMATION")
    report("=" * 70)

    report(f"\nDataset path : {DATASET_PATH.relative_to(PROJECT_ROOT)}")
    report(f"Rows         : {df.shape[0]}")
    report(f"Columns      : {df.shape[1]}")
    report(f"Feature cols : {df.shape[1] - 1}  (all numeric, float64)")
    report(f"Target col   : 'Diagnosis'  (string labels)")

    report("\n--- Column Names and Data Types ---")
    for i, col in enumerate(df.columns):
        report(f"  [{i:2d}] {col:<10}  dtype={df[col].dtype}")

    report("\n--- First 5 Rows (sample) ---")
    report(df.head().to_string())

    report("\n--- Last 3 Rows (sample) ---")
    report(df.tail(3).to_string())


# ===========================================================================
# SECTION 3: Missing Values
# ===========================================================================

def check_missing_values(df: pd.DataFrame):
    """Count NaN/null values in each column."""
    report("\n" + "=" * 70)
    report("SECTION 2: MISSING VALUES")
    report("=" * 70)

    missing = df.isnull().sum()
    total_missing = missing.sum()

    if total_missing == 0:
        report("\n  No missing (NaN) values found in any column.")
        report("  NOTE: This does NOT mean the data is complete.")
        report("  See Section 7 for the LYMp/LYMn pre-imputation finding.")
    else:
        report(f"\n  Total missing values: {total_missing}")
        for col, n in missing[missing > 0].items():
            pct = n / len(df) * 100
            report(f"  {col:<10} : {n} missing  ({pct:.1f}%)")


# ===========================================================================
# SECTION 4: Duplicate Rows
# ===========================================================================

def check_duplicates(df: pd.DataFrame):
    """Count exact duplicate rows and show their class distribution."""
    report("\n" + "=" * 70)
    report("SECTION 3: DUPLICATE ROW ANALYSIS")
    report("=" * 70)

    n_dup      = df.duplicated().sum()
    n_involved = df[df.duplicated(keep=False)].shape[0]

    report(f"\n  Extra duplicate copies (to be removed before training): {n_dup}")
    report(f"  Total rows involved in duplicates (incl. originals)   : {n_involved}")
    report(f"  Rows retained after removing duplicates               : {len(df) - n_dup}")

    report("\n  Duplicates by Diagnosis class:")
    dup_frame = df[df.duplicated(keep=False)]
    by_class  = dup_frame.groupby("Diagnosis").size().sort_values(ascending=False)
    for label, count in by_class.items():
        report(f"    {label:<45} {count:>3} rows involved")

    report("\n  CLEANING DECISION (approved in Phase 1):")
    report("  - Remove 49 exact duplicate rows BEFORE the train/test split.")
    report("  - Work on a cleaned in-memory copy. Original CSV is untouched.")
    report("  - Reason: duplicates across the split inflate test accuracy")
    report("    by letting the model 'remember' training rows at test time.")


# ===========================================================================
# SECTION 5: Class Distribution
# ===========================================================================

def check_class_distribution(df: pd.DataFrame):
    """Show the count and percentage of each diagnosis class."""
    report("\n" + "=" * 70)
    report("SECTION 4: DIAGNOSIS CLASS DISTRIBUTION")
    report("=" * 70)

    dist  = df["Diagnosis"].value_counts(dropna=False)
    total = len(df)

    report(f"\n  {'Diagnosis':<45} {'Count':>6}  {'%':>6}")
    report(f"  {'-' * 45} {'-' * 6}  {'-' * 6}")
    for label, count in dist.items():
        pct = count / total * 100
        bar = "=" * int(pct / 2)
        report(f"  {label:<45} {count:>6}  {pct:>5.1f}%  {bar}")

    report(f"\n  Total records: {total}")
    report("\n  IMBALANCE NOTES:")
    report("  - Top 3 classes account for ~69% of records.")
    report("  - Macrocytic anemia (18 rows) and Leukemia+thrombocytopenia (11 rows)")
    report("    are very rare. Reliable per-class metrics for these are not possible.")
    report("  - We will use class_weight='balanced' where applicable and report")
    report("    macro F1 (all classes equal weight) alongside weighted F1.")


# ===========================================================================
# SECTION 6: Descriptive Statistics
# ===========================================================================

def descriptive_statistics(df: pd.DataFrame):
    """Show min, max, mean, std, quartiles for all 14 numeric features."""
    report("\n" + "=" * 70)
    report("SECTION 5: DESCRIPTIVE STATISTICS (numeric features)")
    report("=" * 70)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    desc = df[numeric_cols].describe()
    report("\n" + desc.to_string())

    report("\n  HOW TO READ THIS TABLE:")
    report("  - count : number of non-null values  (1281 = no NaN)")
    report("  - mean  : average across all patients")
    report("  - std   : how spread out values are (larger = more variation)")
    report("  - min   : smallest recorded value")
    report("  - 25%   : value below which 25% of data falls (Q1)")
    report("  - 50%   : median — the middle value")
    report("  - 75%   : value below which 75% of data falls (Q3)")
    report("  - max   : largest recorded value")


# ===========================================================================
# SECTION 7: Suspicious / Impossible Values
# ===========================================================================

# Clinical bounds used ONLY for flagging — NOT for automatic deletion.
# These are intentionally wider than strict normal to avoid false alarms.
# Values outside these limits need human review and investigation.
# Source: standard CBC adult reference interpretation (approximate).
CLINICAL_BOUNDS = {
    # column : (lower, upper, unit, descriptive note)
    "WBC"  : (0.5,   50.0,  "x10^3/uL", "very low or very high WBC"),
    "LYMp" : (5.0,   95.0,  "%",         "lymphocyte % (max is 100%)"),
    "NEUTp": (0.5,  100.0,  "%",         "neutrophil % (max is 100%)"),
    "LYMn" : (0.05, 20.0,   "x10^3/uL", "absolute lymphocyte count"),
    "NEUTn": (0.1,  50.0,   "x10^3/uL", "absolute neutrophil count"),
    "RBC"  : (1.0,  10.0,   "x10^6/uL", "red blood cell count"),
    "HGB"  : (0.0,  25.0,   "g/dL",      "haemoglobin (cannot be negative)"),
    "HCT"  : (5.0,  75.0,   "%",         "haematocrit (cannot exceed 100%)"),
    "MCV"  : (30.0, 150.0,  "fL",        "mean corpuscular volume"),
    "MCH"  : (10.0,  50.0,  "pg",        "mean corpuscular haemoglobin"),
    "MCHC" : (15.0,  45.0,  "g/dL",     "mean corpuscular HGB concentration"),
    "PLT"  : (5.0,  700.0,  "x10^3/uL", "platelet count"),
    "PDW"  : (7.0,   40.0,  "fL",        "platelet distribution width"),
    "PCT"  : (0.005,  5.0,  "%",         "plateletcrit"),
}


def check_suspicious_values(df: pd.DataFrame):
    """
    Flag rows with values outside documented clinical bounds.
    Does NOT delete or modify any rows — only reports them for review.
    """
    report("\n  Clinical bounds used for flagging are intentionally wide.")
    report("  Values outside these limits need human review.")
    report("  These rows are NOT deleted — listed here for investigation.\n")

    total_suspicious_rows = set()

    for col, (lo, hi, unit, note) in CLINICAL_BOUNDS.items():
        if col not in df.columns:
            continue

        flagged_low  = df[df[col] < lo]
        flagged_high = df[df[col] > hi]
        n_low  = len(flagged_low)
        n_high = len(flagged_high)
        n_total = n_low + n_high

        if n_total == 0:
            continue

        report(f"  >> {col} ({unit}) — expected range [{lo}, {hi}]")
        report(f"     Note: {note}")
        report(f"     Values below {lo}: {n_low}  |  Values above {hi}: {n_high}  "
               f"|  Total flagged: {n_total}")
        report(f"     Data range: min={df[col].min():.3f}, max={df[col].max():.3f}")

        all_flagged = pd.concat([flagged_low, flagged_high]).drop_duplicates()
        total_suspicious_rows.update(all_flagged.index.tolist())

        # Show the most extreme values (up to 3 low, 3 high)
        sample_low  = flagged_low.nsmallest(3, col)  if n_low  > 0 else pd.DataFrame()
        sample_high = flagged_high.nlargest(3, col)   if n_high > 0 else pd.DataFrame()
        sample = pd.concat([sample_low, sample_high]).drop_duplicates()

        for idx, row in sample.iterrows():
            report(f"     Row {idx:4d}: {col}={row[col]:>12.3f}   "
                   f"Diagnosis: {row['Diagnosis']}")
        report()

    report(f"  SUMMARY: {len(total_suspicious_rows)} unique rows contain at least")
    report(f"           one suspicious value across all 14 columns.")
    report()
    report("  NEXT STEPS (Phase 4 — Preprocessing):")
    report("  1. Investigate each flagged row.")
    report("  2. Correct values ONLY when there is reliable evidence")
    report("     (e.g., NEUTp=5317 is clearly not a valid percentage).")
    report("  3. If a value cannot be verified, document the exclusion decision.")
    report("  4. Run a sensitivity analysis: compare model with vs without")
    report("     the rows containing extreme values.")


# ===========================================================================
# SECTION 8: LYMp and LYMn Pre-Imputation Flag
# ===========================================================================

def check_preimputed_columns(df: pd.DataFrame):
    """
    Investigate LYMp and LYMn for suspected hidden pre-imputation.
    61% of rows share the exact same value (the column median) in these columns.
    This pattern is characteristic of median-fill imputation done before publishing.
    """
    report("\n" + "=" * 70)
    report("SECTION 7: LYMp and LYMn — SUSPECTED PRE-IMPUTATION WARNING")
    report("=" * 70)

    for col in ["LYMp", "LYMn"]:
        if col not in df.columns:
            continue
        mode_val   = df[col].mode()[0]
        mode_count = (df[col] == mode_val).sum()
        pct        = mode_count / len(df) * 100
        col_median = df[col].median()

        report(f"\n  {col}:")
        report(f"    Most common value (mode) : {mode_val}")
        report(f"    Column median            : {col_median}")
        report(f"    Unique values            : {df[col].nunique()}")
        report(f"    Rows with mode value     : {mode_count} / {len(df)} ({pct:.1f}%)")

        if abs(mode_val - col_median) < 1e-6:
            report(f"    ** WARNING: The most common value IS the column median.")
            report(f"       This strongly suggests missing values were pre-filled")
            report(f"       with the median before the CSV was published.")
            report(f"       The real missing-data signal is lost for {pct:.0f}% of rows.")

    report("\n  DECISION (approved in Phase 1):")
    report("  - Keep both features. Do NOT remove or replace their values.")
    report("  - Document this warning in all reports and the project README.")
    report("  - In Phase 5, compare model with and without LYMp/LYMn to")
    report("    measure how much they contribute.")


# ===========================================================================
# SECTION 9: Generate and Save Plots
# ===========================================================================

def save_diagnosis_distribution_plot(df: pd.DataFrame):
    """
    Save a horizontal bar chart of diagnosis class counts.
    Output: ml_model/reports/diagnosis_distribution.png
    """
    dist   = df["Diagnosis"].value_counts()
    labels = dist.index.tolist()
    counts = dist.values.tolist()
    total  = len(df)

    # Build a colour palette: most common class = darkest blue
    palette = sns.color_palette("Blues_d", n_colors=len(labels))

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(
        y=list(reversed(labels)),
        width=list(reversed(counts)),
        color=list(reversed(palette)),
        edgecolor="white",
        linewidth=0.8,
    )

    # Annotate each bar with count + percentage
    for bar, count in zip(bars, list(reversed(counts))):
        pct = count / total * 100
        ax.text(
            bar.get_width() + 4,
            bar.get_y() + bar.get_height() / 2,
            f"{count}  ({pct:.1f}%)",
            va="center", ha="left", fontsize=9, color="#333333",
        )

    ax.set_xlabel("Number of Records", fontsize=11)
    ax.set_title(
        "Diagnosis Class Distribution\n"
        f"Total: {total} records  |  9 classes  |  Source: anemia_types.csv",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax.set_xlim(0, max(counts) * 1.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    out_path = REPORTS_DIR / "diagnosis_distribution.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def save_feature_boxplots(df: pd.DataFrame):
    """
    Save boxplots for all 14 numeric features side by side.
    Boxplots reveal spread, median, and extreme outliers at a glance.
    Output: ml_model/reports/feature_boxplots.png
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    n_cols = len(numeric_cols)
    n_rows = (n_cols + 1) // 2  # 2 subplots per row

    fig, axes = plt.subplots(n_rows, 2, figsize=(14, n_rows * 3))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        ax.boxplot(
            df[col].dropna(),
            orientation="vertical",
            patch_artist=True,
            boxprops=dict(facecolor="#AED6F1", color="#1A5276"),
            medianprops=dict(color="#E74C3C", linewidth=2),
            flierprops=dict(marker="o", markerfacecolor="#E74C3C",
                            markersize=3, alpha=0.5),
            whiskerprops=dict(color="#1A5276"),
            capprops=dict(color="#1A5276"),
        )
        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused subplot slots
    for j in range(n_cols, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "CBC Feature Distributions — Boxplots (14 parameters)\n"
        "Red dots = outliers  |  Red line = median  |  Blue box = Q1–Q3 range",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out_path = REPORTS_DIR / "feature_boxplots.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


# ===========================================================================
# SECTION 10: Save Text Report
# ===========================================================================

def save_text_report():
    """Write all collected report lines to a plain-text file."""
    out_path = REPORTS_DIR / "analysis_report.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(_report_lines))
    print(f"  Text report saved  -> {out_path.relative_to(PROJECT_ROOT)}")
    return out_path


# ===========================================================================
# MAIN ENTRY POINT
# ===========================================================================

def main():
    report("=" * 70)
    report("AI-Based Anemia Screening — Phase 3: Exploratory Data Analysis")
    report("Educational project — NOT a clinical diagnostic tool.")
    report("=" * 70)

    # 1. Load dataset
    report(f"\nLoading dataset from:\n  {DATASET_PATH.relative_to(PROJECT_ROOT)}\n")
    df = load_dataset(DATASET_PATH)
    report(f"Dataset loaded successfully.  Shape: {df.shape}\n")

    # 2. Run all analysis sections
    print_basic_info(df)
    check_missing_values(df)
    check_duplicates(df)
    check_class_distribution(df)
    descriptive_statistics(df)

    report("\n" + "=" * 70)
    report("SECTION 6: SUSPICIOUS / IMPOSSIBLE VALUE INVESTIGATION")
    report("=" * 70)
    check_suspicious_values(df)

    check_preimputed_columns(df)

    # 3. Generate plots
    report("\n" + "=" * 70)
    report("SECTION 8: SAVING PLOTS TO ml_model/reports/")
    report("=" * 70)
    p1 = save_diagnosis_distribution_plot(df)
    report(f"  Saved -> {p1.relative_to(PROJECT_ROOT)}")
    p2 = save_feature_boxplots(df)
    report(f"  Saved -> {p2.relative_to(PROJECT_ROOT)}")

    # 4. Final summary
    report("\n" + "=" * 70)
    report("ANALYSIS COMPLETE — SUMMARY")
    report("=" * 70)
    report(f"\n  Total records          : {len(df)}")
    report(f"  Diagnosis classes      : {df['Diagnosis'].nunique()}")
    report(f"  Missing values (NaN)   : 0  (but see LYMp/LYMn warning)")
    report(f"  Exact duplicate rows   : {df.duplicated().sum()} (remove before training)")
    report(f"  Records after dedup    : {len(df) - df.duplicated().sum()}")
    report(f"\n  Files saved in        : ml_model/reports/")
    report(f"    diagnosis_distribution.png")
    report(f"    feature_boxplots.png")
    report(f"    analysis_report.txt")
    report()
    report("  NEXT STEP: Phase 4 — Implement preprocessing.py")
    report("  (Await review of these findings before proceeding)")
    report()
    report("  DISCLAIMER: This is an educational screening-support tool.")
    report("  Do not use model outputs for clinical decisions.")
    report("  Consult a qualified healthcare professional for interpretation.")

    # 5. Save the text report
    report("\n" + "=" * 70)
    save_text_report()


if __name__ == "__main__":
    main()

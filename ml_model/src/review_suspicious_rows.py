"""
review_suspicious_rows.py — Suspicious Record Review
====================================================
AI-Based Anemia Screening — Supporting script for Phase 3/4 transition.

PURPOSE
-------
Identifies and documents every flagged record from the dataset.
Does NOT modify, delete, or cap any values.
Does NOT touch the original CSV.

Outputs:
  ml_model/reports/suspicious_rows_review.csv  — machine-readable record per flagged row
  ml_model/reports/suspicious_rows_review.txt  — human-readable detailed report

HOW TO RUN (from project root):
    source .venv/bin/activate
    python ml_model/src/review_suspicious_rows.py
"""

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
from pathlib import Path

# ─── Paths ──────────────────────────────────────────────────────────────────
THIS_FILE    = Path(__file__).resolve()
ML_DIR       = THIS_FILE.parent.parent      # ml_model/
PROJECT_ROOT = ML_DIR.parent                # AI-Anemia-Screening/
DATASET_PATH = ML_DIR / "dataset" / "anemia_types.csv"
REPORTS_DIR  = ML_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Clinical bounds (same as in analyze_data.py) ───────────────────────────
# These are intentionally wide to catch clearly impossible values
# without over-flagging plausible extremes.
# (lower_limit, upper_limit, unit, description)
CLINICAL_BOUNDS = {
    "WBC"  : (0.5,   50.0,  "x10^3/uL", "White blood cell count"),
    "LYMp" : (5.0,   95.0,  "%",         "Lymphocyte percentage"),
    "NEUTp": (0.5,  100.0,  "%",         "Neutrophil percentage — cannot exceed 100"),
    "LYMn" : (0.05, 20.0,   "x10^3/uL", "Absolute lymphocyte count"),
    "NEUTn": (0.1,  50.0,   "x10^3/uL", "Absolute neutrophil count"),
    "RBC"  : (1.0,  10.0,   "x10^6/uL", "Red blood cell count"),
    "HGB"  : (0.0,  25.0,   "g/dL",      "Haemoglobin"),
    "HCT"  : (5.0,  75.0,   "%",         "Haematocrit — cannot exceed 100"),
    "MCV"  : (30.0, 150.0,  "fL",        "Mean corpuscular volume"),
    "MCH"  : (10.0,  50.0,  "pg",        "Mean corpuscular haemoglobin"),
    "MCHC" : (15.0,  45.0,  "g/dL",      "Mean corpuscular HGB concentration"),
    "PLT"  : (5.0,  700.0,  "x10^3/uL", "Platelet count"),
    "PDW"  : (7.0,   40.0,  "fL",        "Platelet distribution width"),
    "PCT"  : (0.005,  5.0,  "%",         "Plateletcrit"),
}

# ─── Category rules ─────────────────────────────────────────────────────────
# Applied per (column, value) pair.
# Returns: (category letter, brief rationale)
#
# A — Demonstrably invalid: reliable physical/logical evidence makes the
#     value impossible (e.g., negative concentration, % > 100).
#
# B — Suspicious / unverifiable: the value is extreme and likely erroneous
#     but we cannot confirm the cause from this dataset alone.
#     (e.g., MCH=275 could be a ×10 unit error, but we cannot prove it.)
#
# C — Extreme but possibly genuine: the value falls outside the flagging
#     bounds but has a plausible pathological explanation.

def classify_flag(col: str, value: float) -> tuple[str, str]:
    """
    Return (category, rationale) for a single flagged (column, value) pair.
    Category is one of 'A', 'B', or 'C'.
    """

    # ── CATEGORY A: physically/logically impossible ──────────────────────
    # Haemoglobin cannot be negative
    if col == "HGB" and value < 0:
        return ("A",
                "HGB is negative. Haemoglobin concentration cannot be "
                "less than 0 g/dL. This is physically impossible.")

    # Percentage features cannot exceed 100
    if col == "NEUTp" and value > 100:
        return ("A",
                f"NEUTp={value:.1f}. Neutrophil percentage cannot exceed "
                "100%. This is likely a data-entry error (possibly raw "
                "count entered instead of percentage).")

    if col == "HCT" and value > 100:
        return ("A",
                f"HCT={value:.1f}. Haematocrit is a percentage; values "
                "above 100% are physically impossible. Likely a data-entry "
                "or unit error (e.g., multiplied by 10 or 100).")

    # Volume cannot be negative
    if col == "MCV" and value < 0:
        return ("A",
                f"MCV={value:.1f} fL. Cell volume cannot be negative. "
                "This is a data-entry error.")

    # MCH values > 500 pg are ~15–100× the physiological maximum (~33 pg).
    # These are so extreme that no known disease produces them.
    if col == "MCH" and value > 500:
        return ("A",
                f"MCH={value:.1f} pg. The physiological maximum is ~45 pg. "
                "Values like 2511 or 3117 are 75–95× the maximum. "
                "This is almost certainly a data-entry error (e.g., extra digits).")

    # ── CATEGORY B: suspicious, cause unverifiable from this dataset ─────

    # RBC > 15 x10^6/uL — the highest ever recorded in polycythaemia is ~8–9
    if col == "RBC" and value > 15.0:
        return ("B",
                f"RBC={value:.2f} x10^6/uL. The maximum physiologically "
                "plausible value is ~9–10. Values like 29.7, 41.1, 90.8 "
                "are likely unit errors (e.g., recorded as x10^4 instead "
                "of x10^6) but this cannot be verified from the CSV alone.")

    # HGB > 25 g/dL — highest ever recorded is ~23 g/dL
    if col == "HGB" and value > 25:
        return ("B",
                f"HGB={value:.1f} g/dL. The highest reliably reported HGB "
                "is ~23 g/dL. Values of 41, 85, 87 are likely data-entry "
                "or unit errors but cannot be confirmed from this dataset.")

    # MCH 50–499 range — 50–100× normal maximum is suspicious but not
    # impossible to explain as a unit error; 100–500 is increasingly unlikely
    if col == "MCH" and value > 50:
        return ("B",
                f"MCH={value:.1f} pg. Normal range is 27–33 pg; upper "
                "clinical limit ~45 pg. Values like 275 are ~10× the "
                "maximum. Likely a ×10 unit error, but unverifiable here.")

    # MCHC > 40 g/dL — values above 40 are very unusual; above 50 virtually
    # never occur in human blood
    if col == "MCHC" and value > 40:
        return ("B",
                f"MCHC={value:.1f} g/dL. Values above 40 are clinically "
                "rare (hyperchromia); above 50 virtually never occur "
                "physiologically. Likely a data-entry error, but "
                "unverifiable from this dataset.")

    # MCV > 150 fL — the highest seen in macrocytic anaemia is ~130 fL;
    # MCV=990 is clearly wrong
    if col == "MCV" and value > 150:
        return ("B",
                f"MCV={value:.1f} fL. Clinical maximum is ~130 fL even "
                "in severe macrocytosis. This appears to be a data-entry "
                "error, but the correct value cannot be inferred here.")

    # LYMn > 20 x10^3/uL — extremely high but conceivable in CLL/ALL;
    # however LYMn=41.8 is borderline
    if col == "LYMn" and value > 20:
        return ("B",
                f"LYMn={value:.3f} x10^3/uL. Extreme lymphocytosis "
                "(>20) can occur in CLL but values like 41.8 are rare. "
                "Cannot confirm or rule out from this dataset alone.")

    # NEUTn > 50 x10^3/uL
    if col == "NEUTn" and value > 50:
        return ("B",
                f"NEUTn={value:.1f} x10^3/uL. Extreme neutrophilia "
                "(>50) is possible in severe infection but very unusual. "
                "Value of 79 with the associated diagnosis needs review.")

    # PCT > 5%
    if col == "PCT" and value > 5:
        return ("B",
                f"PCT={value:.1f}%. Plateletcrit above 5% is extreme. "
                "Typical range is 0.1–0.5%. These values (11–14%) are "
                "likely data-entry errors but cannot be confirmed.")

    # PDW > 40 fL
    if col == "PDW" and value > 40:
        return ("B",
                f"PDW={value:.1f} fL. PDW above 40 is highly unusual. "
                "PDW=97 fL appears to be a data-entry error but the "
                "correct value cannot be recovered from this dataset.")

    # LYMp outliers — high values
    if col == "LYMp" and value > 80:
        return ("B",
                f"LYMp={value:.1f}%. Values above 80% are possible in "
                "viral infections or lymphocytic leukaemia but 89–91% "
                "is extreme and worth review.")

    # ── CATEGORY C: extreme but possibly genuine ─────────────────────────

    # Low WBC — neutropenia/agranulocytosis can produce very low WBC
    if col == "WBC" and value < 1.0:
        return ("C",
                f"WBC={value:.2f} x10^3/uL. Very low but possible in "
                "severe neutropenia, bone marrow failure, or chemotherapy. "
                "No strong evidence this is an error.")

    # Low MCV — severe microcytic anaemia can produce MCV as low as 50–55 fL;
    # values like 25–47 fL are unusual but not physically impossible
    if col == "MCV" and value < 30:
        return ("C",
                f"MCV={value:.1f} fL. Extremely low; could represent "
                "severe microcytosis or thalassaemia. Unusual but not "
                "physically impossible. Keep and investigate.")

    # Low MCH
    if col == "MCH" and value < 15:
        return ("C",
                f"MCH={value:.1f} pg. Very low but possible in severe "
                "microcytic/hypochromic anaemia. Not physically impossible.")

    # Low MCHC
    if col == "MCHC" and value < 20:
        return ("C",
                f"MCHC={value:.1f} g/dL. Very low; possible in severe "
                "hypochromic anaemia. No direct evidence of an error.")

    # Slightly elevated LYMp (possible in viral illness)
    if col == "LYMp" and value < 10:
        return ("C",
                f"LYMp={value:.1f}%. Low lymphocyte % is seen in "
                "neutrophilia or steroid treatment. Unusual but plausible.")

    # NEUTp slightly below lower bound
    if col == "NEUTp" and value < 0.5:
        return ("C",
                f"NEUTp={value:.1f}%. Very low neutrophil % could occur "
                "in severe lymphocytosis. Unusual but not impossible.")

    # PLT slightly above upper bound
    if col == "PLT" and value > 700:
        return ("C",
                f"PLT={value:.0f} x10^3/uL. Slightly above 700 threshold. "
                "Reactive thrombocytosis can produce PLT up to ~800–1000. "
                "A single value of 660 is borderline, not clearly wrong.")

    # Fallback — anything else flagged but not matched above
    return ("B",
            f"{col}={value} is outside expected range [{CLINICAL_BOUNDS[col][0]}, "
            f"{CLINICAL_BOUNDS[col][1]}] {CLINICAL_BOUNDS[col][2]}. "
            "Classification requires further domain investigation.")


# ─── Main review logic ───────────────────────────────────────────────────────

def build_review_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each row that has at least one flagged value, build a record
    describing all triggered flags, their categories, and the full
    set of CBC values.
    """
    feature_cols = [c for c in df.columns if c != "Diagnosis"]
    records = []

    # Track which rows have been flagged (using df index)
    flagged_indices = set()
    for col, (lo, hi, unit, _) in CLINICAL_BOUNDS.items():
        if col not in df.columns:
            continue
        flagged_indices.update(df[df[col] < lo].index.tolist())
        flagged_indices.update(df[df[col] > hi].index.tolist())

    for df_idx in sorted(flagged_indices):
        row = df.loc[df_idx]

        # CSV row number: index 0 = row 2 in CSV (row 1 is the header)
        csv_row = df_idx + 2

        # Collect all flags for this row
        row_flags = []
        categories_seen = []
        for col, (lo, hi, unit, desc) in CLINICAL_BOUNDS.items():
            if col not in df.columns:
                continue
            val = row[col]
            if val < lo or val > hi:
                direction = "below minimum" if val < lo else "above maximum"
                cat, rationale = classify_flag(col, val)
                categories_seen.append(cat)
                row_flags.append({
                    "column"    : col,
                    "value"     : val,
                    "unit"      : unit,
                    "bound_lo"  : lo,
                    "bound_hi"  : hi,
                    "direction" : direction,
                    "category"  : cat,
                    "rationale" : rationale,
                })

        # The overall row category = worst category (A > B > C)
        if "A" in categories_seen:
            row_cat = "A"
        elif "B" in categories_seen:
            row_cat = "B"
        else:
            row_cat = "C"

        flagged_cols_str  = "; ".join(f["column"] for f in row_flags)
        flag_details_str  = " | ".join(
            f"{f['column']}={f['value']} ({f['direction']})" for f in row_flags
        )
        rationale_str     = " | ".join(f["rationale"] for f in row_flags)

        rec = {
            "csv_row_number" : csv_row,
            "df_index"       : df_idx,
            "Diagnosis"      : row["Diagnosis"],
            **{col: row[col] for col in feature_cols},
            "flagged_columns": flagged_cols_str,
            "flag_details"   : flag_details_str,
            "row_category"   : row_cat,
            "rationale"      : rationale_str,
            "_flags"         : row_flags,   # internal only, not saved to CSV
        }
        records.append(rec)

    return records


def save_csv_report(records: list, out_path: Path):
    """Save a flat CSV with one row per suspicious record."""
    rows_for_csv = []
    for r in records:
        row = {k: v for k, v in r.items() if k != "_flags"}
        rows_for_csv.append(row)
    df_out = pd.DataFrame(rows_for_csv)
    df_out.to_csv(out_path, index=False)
    print(f"  CSV report saved  -> {out_path.relative_to(PROJECT_ROOT)}")


def save_text_report(records: list, df_original: pd.DataFrame, out_path: Path):
    """Save a human-readable detailed text report."""
    lines = []

    def L(text=""):
        lines.append(str(text))

    L("=" * 78)
    L("SUSPICIOUS RECORD REVIEW — AI-Based Anemia Screening Project")
    L("Phase 3/4 Transition — Pre-Preprocessing Investigation")
    L("=" * 78)
    L()
    L("IMPORTANT NOTES")
    L("-" * 78)
    L("1. This report does NOT modify, delete, or cap any values.")
    L("2. The original CSV is untouched (MD5 verified before and after).")
    L("3. Categories are assigned per (column, value) pair:")
    L("   A — Demonstrably invalid: reliable physical or logical evidence")
    L("       proves the value is impossible (e.g., HGB < 0, NEUTp > 100%).")
    L("   B — Suspicious / unverifiable: the value is extreme and likely")
    L("       erroneous but the cause cannot be established from this data.")
    L("   C — Extreme but possibly genuine: unusual value with a plausible")
    L("       pathological explanation; no strong evidence of an error.")
    L()
    L("TERMINOLOGY")
    L("-" * 78)
    L("  CSV Row Number : 1-indexed line in anemia_types.csv (row 1 = header)")
    L("  DataFrame Index: 0-indexed row position in pandas (0 = first data row)")
    L("  Relationship   : CSV row = DataFrame index + 2")
    L()
    L(f"TOTAL FLAGGED: {len(records)} unique rows contain at least one suspicious value.")
    L()
    L("=" * 78)
    L("DETAILED ROW-BY-ROW REVIEW")
    L("=" * 78)

    cat_counts = {"A": 0, "B": 0, "C": 0}

    for i, rec in enumerate(records, start=1):
        cat_counts[rec["row_category"]] += 1
        flags = rec["_flags"]

        L()
        L(f"{'─' * 78}")
        L(f"  RECORD {i:2d} of {len(records)}  |  "
          f"CSV Row: {rec['csv_row_number']}  |  "
          f"DataFrame Index: {rec['df_index']}  |  "
          f"Overall Category: {rec['row_category']}")
        L(f"  Diagnosis: {rec['Diagnosis']}")
        L(f"{'─' * 78}")

        # All 14 CBC values
        feature_cols = ["WBC","LYMp","NEUTp","LYMn","NEUTn",
                        "RBC","HGB","HCT","MCV","MCH","MCHC","PLT","PDW","PCT"]
        L()
        L("  ALL 14 CBC FEATURE VALUES:")
        for col in feature_cols:
            unit = CLINICAL_BOUNDS.get(col, ("","","",""))[2]
            val  = rec[col]
            is_flagged = any(f["column"] == col for f in flags)
            flag_marker = " <-- FLAGGED" if is_flagged else ""
            L(f"    {col:<8} = {val:>12.4f}  {unit:<12}{flag_marker}")

        L()
        L("  FLAG DETAILS:")
        for f in flags:
            L(f"    Column  : {f['column']} ({f['unit']})")
            L(f"    Value   : {f['value']}")
            L(f"    Bounds  : [{f['bound_lo']}, {f['bound_hi']}]")
            L(f"    Issue   : {f['direction']}")
            L(f"    Category: {f['category']}")
            L(f"    Rationale: {f['rationale']}")
            L()

    L()
    L("=" * 78)
    L("SUMMARY BY CATEGORY")
    L("=" * 78)
    L()
    L(f"  Category A (Demonstrably invalid)   : {cat_counts['A']:2d} rows")
    L(f"  Category B (Suspicious/unverifiable): {cat_counts['B']:2d} rows")
    L(f"  Category C (Extreme but plausible)  : {cat_counts['C']:2d} rows")
    L(f"  Total flagged rows                  : {len(records):2d}")
    L()
    L("  (Rows can appear in only one category — the worst flag determines")
    L("   the row's category. A row with one A-flag and one C-flag = Category A.)")
    L()
    L("=" * 78)
    L("PROPOSED HANDLING POLICY")
    L("=" * 78)
    L()
    L("CATEGORY A — Demonstrably Invalid")
    L("  These rows contain values that are physically or logically impossible.")
    L("  The correct value cannot be recovered from the CSV alone.")
    L("  Proposed action: EXCLUDE from model training with full documentation.")
    L("  Exclusion criteria:")
    L("    - HGB < 0 (negative haemoglobin)")
    L("    - NEUTp > 100 (percentage exceeds 100%)")
    L("    - HCT > 100 (percentage exceeds 100%)")
    L("    - MCV < 0 (negative cell volume)")
    L("    - MCH > 500 (100× physiological maximum)")
    L()
    L("CATEGORY B — Suspicious / Unverifiable")
    L("  These rows have extreme values that are likely wrong, but the")
    L("  original correct value cannot be established from this dataset.")
    L("  Proposed action: RETAIN in Scenario 1 (conservative).")
    L("                   EXCLUDE in Scenario 2 (aggressive) for sensitivity.")
    L()
    L("CATEGORY C — Extreme but Possibly Genuine")
    L("  These rows have values outside the flagging bounds but within")
    L("  plausible pathological ranges. No reliable evidence of error.")
    L("  Proposed action: RETAIN in both scenarios.")
    L()
    L("=" * 78)
    L("PREPROCESSING SENSITIVITY ANALYSIS SCENARIOS")
    L("=" * 78)
    L()
    L("Scenario 1 — Conservative (baseline)")
    L("  - Remove 49 exact duplicate rows (all scenarios).")
    L("  - Exclude ONLY Category A rows (demonstrably invalid values).")
    L("  - Retain all Category B and C rows.")
    L("  - Document counts for each task (A multiclass, B binary screening).")
    L()
    L("Scenario 2 — Aggressive exclusion (sensitivity test)")
    L("  - Remove 49 exact duplicate rows (all scenarios).")
    L("  - Exclude Category A rows (same as Scenario 1).")
    L("  - ALSO exclude Category B rows (suspicious, unverifiable).")
    L("  - Retain only Category C rows alongside clean data.")
    L("  - Document counts for each task.")
    L("  - Train identical models on both scenarios and compare performance.")
    L("  - If results are substantially different, report both honestly.")
    L("  - If results are similar, the suspicious rows are not driving outcomes.")
    L()
    L("IMPORTANT: Train/test split happens AFTER all row-level exclusions.")
    L("           No test-set information is used to decide exclusions.")
    L("           Exclusion decisions are based solely on value impossibility,")
    L("           not on model performance or test accuracy.")
    L()
    L("=" * 78)
    L("SAMPLE COUNTS AFTER EACH SCENARIO (estimated, before split)")
    L("=" * 78)
    L()

    # Compute actual counts
    n_original = len(df_original)
    n_after_dedup = n_original - df_original.duplicated().sum()
    cat_a_indices = set()
    cat_b_indices = set()
    for rec in records:
        if rec["row_category"] == "A":
            cat_a_indices.add(rec["df_index"])
        elif rec["row_category"] == "B":
            cat_b_indices.add(rec["df_index"])

    # After dedup, how many A/B rows remain?
    df_deduped = df_original.drop_duplicates()
    cat_a_in_deduped = cat_a_indices & set(df_deduped.index)
    cat_b_in_deduped = cat_b_indices & set(df_deduped.index)

    n_s1 = n_after_dedup - len(cat_a_in_deduped)
    n_s2 = n_after_dedup - len(cat_a_in_deduped) - len(cat_b_in_deduped)

    L(f"  Original dataset                        : {n_original} rows")
    L(f"  After removing 49 duplicates            : {n_after_dedup} rows")
    L(f"  Category A rows (in deduped set)        : {len(cat_a_in_deduped)} rows")
    L(f"  Category B rows (in deduped set)        : {len(cat_b_in_deduped)} rows")
    L()
    L(f"  Scenario 1 (remove duplicates + Cat-A)  : ~{n_s1} rows")
    L(f"  Scenario 2 (remove duplicates + Cat-A+B): ~{n_s2} rows")
    L()
    L("  For Task B (binary screening), an additional ~131 rows are")
    L("  excluded (Leukemia, Leukemia+thrombocytopenia, Thrombocytopenia).")
    L("  These counts are computed from the Task A dataset above.")
    L()
    L("=" * 78)
    L("END OF SUSPICIOUS RECORD REVIEW REPORT")
    L("=" * 78)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  TXT report saved  -> {out_path.relative_to(PROJECT_ROOT)}")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Suspicious Record Review — AI Anemia Screening Project")
    print("=" * 70)
    print(f"\nLoading: {DATASET_PATH.relative_to(PROJECT_ROOT)}")

    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns. Original CSV not modified.")

    print("\nIdentifying suspicious rows ...")
    records = build_review_table(df)
    print(f"Found {len(records)} unique rows with at least one flagged value.\n")

    csv_out = REPORTS_DIR / "suspicious_rows_review.csv"
    txt_out = REPORTS_DIR / "suspicious_rows_review.txt"
    save_csv_report(records, csv_out)
    save_text_report(records, df, txt_out)

    # Console summary
    print()
    print("=" * 70)
    print("CONSOLE SUMMARY")
    print("=" * 70)
    print()
    print(f"  {'#':<4} {'CSV':>5} {'DFidx':>6}  {'Category':<10} "
          f"{'Diagnosis':<45}  Flagged Columns")
    print(f"  {'-'*4} {'-'*5} {'-'*6}  {'-'*10} {'-'*45}  {'-'*30}")
    for i, rec in enumerate(records, start=1):
        diag = rec["Diagnosis"][:44]
        print(f"  {i:<4} {rec['csv_row_number']:>5} {rec['df_index']:>6}  "
              f"{rec['row_category']:<10} {diag:<45}  {rec['flagged_columns']}")

    cat_counts = {"A": 0, "B": 0, "C": 0}
    for rec in records:
        cat_counts[rec["row_category"]] += 1
    print()
    print(f"  Category A (Demonstrably invalid)   : {cat_counts['A']} rows")
    print(f"  Category B (Suspicious/unverifiable): {cat_counts['B']} rows")
    print(f"  Category C (Extreme but plausible)  : {cat_counts['C']} rows")
    print(f"  Total                               : {len(records)} rows")
    print()
    print("DONE. Awaiting your review before any preprocessing changes.")


if __name__ == "__main__":
    main()

"""
evaluate.py — Phase 5: Evaluation & Model Inspection Utilities
===============================================================
AI-Based Anemia Screening and Blood Report Interpretation System
College Project — Educational Use Only (Not Clinically Validated)

PURPOSE:
--------
Loads saved trained model pipelines from `ml_model/models/`, evaluates them on
the shared clean held-out test partitions, prints detailed diagnostic tables,
and generates performance visualizations.
"""

import sys
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to sys.path
THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parent
ML_DIR = SRC_DIR.parent
PROJECT_ROOT = ML_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_model.src.preprocessing import (
    run_full_preprocessing,
    REPORTS_DIR,
)

MODELS_DIR = ML_DIR / "models"


def load_model(task: str, scenario: str, model_name: str):
    """Load a serialized joblib pipeline."""
    model_path = MODELS_DIR / f"{task}_{scenario}_{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model pipeline not found at: {model_path}")
    return joblib.load(model_path)


def print_evaluation_summary():
    """Print the saved evaluation results table in a clean format."""
    csv_path = REPORTS_DIR / "evaluation_results.csv"
    if not csv_path.exists():
        print("[WARNING] evaluation_results.csv not found. Run train.py first.")
        return

    df = pd.read_csv(csv_path)
    print("\n" + "=" * 80)
    print("PHASE 5: MODEL EVALUATION BENCHMARK SUMMARY (SHARED CLEAN TEST SET)")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)


def print_detailed_per_class(task="task_a", scenario="scenario2", model_name="random_forest"):
    """Print per-class precision and recall from model metadata."""
    meta_path = MODELS_DIR / "model_metadata.json"
    if not meta_path.exists():
        print("[WARNING] model_metadata.json not found. Run train.py first.")
        return

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    key = f"{task}_{scenario}_{model_name}"
    if key not in meta["models"]:
        print(f"Model key {key} not found in metadata.")
        return

    m_info = meta["models"][key]
    print(f"\nDetailed Breakdown for {key.upper()}:")
    print(f"  Test Accuracy: {m_info['test_accuracy']:.4f}")
    if "test_macro_f1" in m_info:
        print(f"  Test Macro F1: {m_info['test_macro_f1']:.4f}")
        print(f"  Test Balanced Acc: {m_info['test_balanced_accuracy']:.4f}")
        print("\n  Per-Class Metrics:")
        for cls, metrics in m_info["per_class"].items():
            print(f"    {cls:<35}: Precision={metrics['precision']:.3f} | Recall={metrics['recall']:.3f}")


if __name__ == "__main__":
    print_evaluation_summary()
    print_detailed_per_class("task_a", "scenario1", "random_forest")
    print_detailed_per_class("task_a", "scenario2", "random_forest")

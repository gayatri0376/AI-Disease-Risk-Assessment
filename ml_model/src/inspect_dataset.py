
import pandas as pd
from pathlib import Path

file_path = Path("ml_model/dataset/anemia_types.csv")

df = pd.read_csv(file_path)

print("\n--- DATASET SHAPE ---")
print(df.shape)

print("\n--- FIRST FIVE ROWS ---")
print(df.head())

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- DUPLICATE ROWS ---")
print(df.duplicated().sum())

print("\n--- POSSIBLE DIAGNOSIS COLUMNS ---")
for column in df.columns:
    if any(word in column.lower()
           for word in ["diagnosis", "class", "label", "anemia", "target"]):
        print(f"\n{column}:")
        print(df[column].value_counts(dropna=False))

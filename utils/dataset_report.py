import pandas as pd

def generate_dataset_report(df):

    report = []

    rows, cols = df.shape

    report.append(f"Dataset contains {rows} rows and {cols} columns.")

    # Missing values
    missing = df.isnull().sum()
    high_missing = missing[missing > 0]

    if len(high_missing) > 0:
        report.append("Columns with missing values detected:")
        for col in high_missing.index:
            pct = round((missing[col] / rows) * 100, 2)
            report.append(f"{col} → {pct}% missing values")

    # High cardinality
    for col in df.columns:
        unique = df[col].nunique()

        if unique > rows * 0.9:
            report.append(f"{col} looks like an ID column (very high unique values)")

    # Skewed features
    numeric = df.select_dtypes(include=["int64", "float64"])

    for col in numeric.columns:
        skew = numeric[col].skew()

        if abs(skew) > 2:
            report.append(f"{col} is highly skewed (skew={round(skew,2)})")

    if len(report) == 1:
        report.append("Dataset looks clean with no major structural issues.")

    return report
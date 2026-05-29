import pandas as pd
import numpy as np
from scipy.stats import skew

def detect_data_issues(df):

    issues = []

    # Missing values
    missing_percent = (df.isnull().sum() / len(df)) * 100

    for col, val in missing_percent.items():
        if val > 50:
            issues.append(f"⚠ Column '{col}' has {round(val,2)}% missing values")

    # Duplicate rows
    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        issues.append(f"⚠ Dataset contains {duplicate_count} duplicate rows")

    # Numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:

        # Outliers detection using IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower) | (df[col] > upper)]

        if len(outliers) > len(df) * 0.05:
            issues.append(f"⚠ Column '{col}' contains many outliers")

        # Skew detection
        if abs(skew(df[col].dropna())) > 2:
            issues.append(f"⚠ Column '{col}' has highly skewed distribution")

    # Possible ID leakage columns
    for col in df.columns:
        if df[col].nunique() == len(df):
            issues.append(f"⚠ Column '{col}' might be an ID column (possible data leakage)")

    return issues
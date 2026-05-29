import pandas as pd

def detect_leakage(df, target):

    warnings = []

    numeric = df.select_dtypes(include=["int64", "float64"])

    if target not in numeric.columns:
        return ["Target column not numeric. Leakage detection skipped."]

    corr = numeric.corr()[target]

    for col, value in corr.items():

        if col != target and abs(value) > 0.9:
            warnings.append(
                f"Column '{col}' has extremely high correlation with target ({round(value,2)}). Possible leakage."
            )

    # identical columns
    for col in df.columns:
        if col != target:
            if df[col].equals(df[target]):
                warnings.append(f"Column '{col}' is identical to target. Severe leakage.")

    if len(warnings) == 0:
        warnings.append("No obvious leakage detected.")

    return warnings
import pandas as pd
import numpy as np
from scipy.stats import skew

def cleaning_suggestions(df):

    suggestions = []

    # Missing value suggestions
    missing_percent = (df.isnull().sum() / len(df)) * 100

    for col, val in missing_percent.items():

        if val > 0 and val < 30:
            suggestions.append(f"💡 Fill missing values in '{col}' using mean or median")

        elif val >= 30 and val < 60:
            suggestions.append(f"💡 Consider advanced imputation for '{col}' (median or model-based)")

        elif val >= 60:
            suggestions.append(f"💡 Column '{col}' has too many missing values — consider dropping it")

    # Numeric skew suggestions
    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:

        s = skew(df[col].dropna())

        if abs(s) > 2:
            suggestions.append(f"💡 Column '{col}' is highly skewed — apply log transformation")

    # Unique column suggestions (possible ID)
    for col in df.columns:

        if df[col].nunique() == len(df):
            suggestions.append(f"💡 Column '{col}' looks like an ID — consider removing it before training")

    return suggestions
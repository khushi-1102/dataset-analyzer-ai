import pandas as pd
import numpy as np


def dataset_health_score(df):

    total_cells = df.shape[0] * df.shape[1]

    missing_values = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()

    missing_ratio = missing_values / total_cells
    duplicate_ratio = duplicate_rows / df.shape[0]

    score = 100

    score -= missing_ratio * 50
    score -= duplicate_ratio * 30

    score = max(0, round(score, 2))

    return {
        "health_score": score,
        "missing_values": int(missing_values),
        "duplicate_rows": int(duplicate_rows),
        "missing_percentage": round(missing_ratio * 100, 2),
        "duplicate_percentage": round(duplicate_ratio * 100, 2)
    }
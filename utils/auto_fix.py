import pandas as pd
from sklearn.preprocessing import LabelEncoder

def auto_fix_dataset(df):

    report = []

    # remove duplicates
    dup = df.duplicated().sum()
    if dup > 0:
        df = df.drop_duplicates()
        report.append(f"Removed {dup} duplicate rows")

    # missing values
    missing_cols = df.columns[df.isnull().any()]

    for col in missing_cols:

        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)
            report.append(f"Filled missing values in {col} using mode")

        else:
            df[col].fillna(df[col].median(), inplace=True)
            report.append(f"Filled missing values in {col} using median")

    # encode categorical
    cat_cols = df.select_dtypes(include=["object"]).columns

    le = LabelEncoder()

    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
        report.append(f"Encoded categorical column {col}")

    return df, report
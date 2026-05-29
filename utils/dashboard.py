import plotly.express as px
import pandas as pd


def missing_values_chart(df):

    missing = df.isnull().sum()

    missing = missing[missing > 0]

    if len(missing) == 0:
        return None

    fig = px.bar(
        x=missing.index,
        y=missing.values,
        labels={"x": "Columns", "y": "Missing Values"},
        title="Missing Values by Column"
    )

    return fig


def feature_distribution(df):

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    if len(numeric_cols) == 0:
        return None

    col = numeric_cols[0]

    fig = px.histogram(
        df,
        x=col,
        title=f"Distribution of {col}"
    )

    return fig


def model_performance_chart(results_df):

    fig = px.bar(
        results_df,
        x="Model",
        y="Accuracy",
        title="Model Performance Comparison"
    )

    return fig
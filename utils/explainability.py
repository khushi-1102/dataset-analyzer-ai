import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

def explain_model(df, target):

    df = df.copy()

    df = df.fillna(0)

    X = df.drop(columns=[target])
    y = df[target]

    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y)

    model = RandomForestClassifier()
    model.fit(X, y)

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    return importance.head(10)
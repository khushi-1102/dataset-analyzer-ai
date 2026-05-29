import pandas as pd
from lazypredict.Supervised import LazyClassifier, LazyRegressor
from sklearn.model_selection import train_test_split


def recommend_model(df, target_column):

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if y.nunique() <= 10:
        model = LazyClassifier(verbose=0)
    else:
        model = LazyRegressor(verbose=0)

    models, predictions = model.fit(X_train, X_test, y_train, y_test)

    return models
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def run_automl(df, target_column):

    df = df.copy()

    # remove missing rows
    df = df.fillna(0)

    # encode categorical features
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(),
        "DecisionTree": DecisionTreeClassifier(),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier()
    }

    results = []

    best_model = None
    best_score = 0

    for name, model in models.items():

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        score = accuracy_score(y_test, preds)

        results.append({
            "Model": name,
            "Accuracy": round(score, 3)
        })

        if score > best_score:
            best_score = score
            best_model = name

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="Accuracy",
        ascending=False
    )

    return best_model, best_score, results_df
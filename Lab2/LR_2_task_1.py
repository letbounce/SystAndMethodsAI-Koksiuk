import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC


def load_income_data(path: str) -> pd.DataFrame:
    columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]
    df = pd.read_csv(
        path,
        header=None,
        names=columns,
        sep=r",\s*",
        engine="python",
        na_values="?",
    )
    return df.dropna().copy()


def build_preprocessor():
    numeric_features = [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]
    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocess


def main():
    lab2_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(lab2_dir, "income_data.txt")

    df = load_income_data(input_path)

    max_per_class = 25_000
    df_le = df[df["income"] == "<=50K"].head(max_per_class)
    df_gt = df[df["income"] == ">50K"].head(max_per_class)
    df_balanced = (
        pd.concat([df_le, df_gt], axis=0)
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )

    X_df = df_balanced.drop(columns=["income"])
    y = df_balanced["income"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=5, stratify=y
    )

    model = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("clf", LinearSVC(random_state=0)),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("=== Task 2.1: Linear SVM results ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision(>50K):", precision_score(y_test, y_pred, pos_label=">50K"))
    print("Recall(>50K):", recall_score(y_test, y_pred, pos_label=">50K"))
    print("F1(>50K):", f1_score(y_test, y_pred, pos_label=">50K"))
    print("F1_weighted:", f1_score(y_test, y_pred, average="weighted"))
    print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred, labels=["<=50K", ">50K"]))
    print("\nClassification report:\n", classification_report(y_test, y_pred))

    cv_f1 = cross_val_score(model, X_df, y, scoring="f1_weighted", cv=3)
    print("\nF1_weighted (CV=3):", cv_f1.mean(), "+/-", cv_f1.std())

    input_data = {
        "age": 37,
        "workclass": "Private",
        "fnlwgt": 215646,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Never-married",
        "occupation": "Handlers-cleaners",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }
    pred = model.predict(pd.DataFrame([input_data]))[0]
    print("\nPredicted class for sample input:", pred)


if __name__ == "__main__":
    main()


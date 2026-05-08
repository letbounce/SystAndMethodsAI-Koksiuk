import os
from collections import Counter, defaultdict
from itertools import product

import numpy as np
import pandas as pd
from sklearn.naive_bayes import CategoricalNB


def load_play_tennis_dataframe() -> pd.DataFrame:
    # Canonical Play Tennis dataset (14 records),
    # consistent with formulas used in the methodological guide.
    data = [
        ("Sunny", "High", "Weak", "No"),
        ("Sunny", "High", "Strong", "No"),
        ("Overcast", "High", "Weak", "Yes"),
        ("Rain", "High", "Weak", "Yes"),
        ("Rain", "Normal", "Weak", "Yes"),
        ("Rain", "Normal", "Strong", "No"),
        ("Overcast", "Normal", "Strong", "Yes"),
        ("Sunny", "High", "Weak", "No"),
        ("Sunny", "Normal", "Weak", "Yes"),
        ("Rain", "Normal", "Weak", "Yes"),
        ("Sunny", "Normal", "Strong", "Yes"),
        ("Overcast", "High", "Strong", "Yes"),
        ("Overcast", "Normal", "Weak", "Yes"),
        ("Rain", "High", "Strong", "No"),
    ]
    return pd.DataFrame(data, columns=["Outlook", "Humidity", "Wind", "Play"])


def frequency_tables(df: pd.DataFrame, features: list[str], target: str) -> dict[str, pd.DataFrame]:
    tables = {}
    for feature in features:
        table = pd.crosstab(df[feature], df[target], margins=True)
        tables[feature] = table
    return tables


def likelihood_tables(df: pd.DataFrame, features: list[str], target: str) -> dict[str, pd.DataFrame]:
    tables = {}
    class_counts = df[target].value_counts().to_dict()
    for feature in features:
        values = sorted(df[feature].unique())
        rows = []
        for val in values:
            row = {"value": val}
            for cls in sorted(df[target].unique()):
                cnt = int(((df[feature] == val) & (df[target] == cls)).sum())
                row[f"P({feature}={val}|{target}={cls})"] = cnt / class_counts[cls]
            rows.append(row)
        tables[feature] = pd.DataFrame(rows)
    return tables


def naive_bayes_posterior_manual(
    df: pd.DataFrame, features: list[str], target: str, evidence: dict[str, str], laplace_alpha: float = 0.0
) -> dict[str, float]:
    classes = sorted(df[target].unique())
    priors = df[target].value_counts(normalize=True).to_dict()

    feature_values = {f: sorted(df[f].unique()) for f in features}
    class_counts = df[target].value_counts().to_dict()

    numerators = {}
    for cls in classes:
        p = priors[cls]
        for f in features:
            val = evidence[f]
            match_count = int(((df[f] == val) & (df[target] == cls)).sum())
            denom = class_counts[cls]
            if laplace_alpha > 0:
                k = len(feature_values[f])
                cond = (match_count + laplace_alpha) / (denom + laplace_alpha * k)
            else:
                cond = match_count / denom
            p *= cond
        numerators[cls] = p

    total = sum(numerators.values())
    return {cls: (numerators[cls] / total if total > 0 else 0.0) for cls in classes}


def fit_sklearn_categorical_nb(
    df: pd.DataFrame, features: list[str], target: str, evidence: dict[str, str]
) -> tuple[dict[str, float], str]:
    encoders: dict[str, dict[str, int]] = {}
    X_encoded_cols = []
    for feature in features:
        values = sorted(df[feature].unique())
        mapping = {v: i for i, v in enumerate(values)}
        encoders[feature] = mapping
        X_encoded_cols.append(df[feature].map(mapping).to_numpy())
    X = np.vstack(X_encoded_cols).T

    y_values = sorted(df[target].unique())
    y_map = {v: i for i, v in enumerate(y_values)}
    y = df[target].map(y_map).to_numpy()

    model = CategoricalNB(alpha=0.0)
    model.fit(X, y)

    x_new = np.array([[encoders[f][evidence[f]] for f in features]])
    proba = model.predict_proba(x_new)[0]
    pred_idx = int(np.argmax(proba))

    result = {label: float(proba[y_map[label]]) for label in y_values}
    pred_label = y_values[pred_idx]
    return result, pred_label


def save_tables_to_markdown(
    freq_tables: dict[str, pd.DataFrame],
    likelihood_tables_dict: dict[str, pd.DataFrame],
    output_md: str,
) -> None:
    lines = ["# Task 2 tables (Play Tennis)\n"]
    lines.append("## Frequency tables\n")
    for feature, table in freq_tables.items():
        lines.append(f"### {feature}\n")
        lines.append(table.to_markdown())
        lines.append("")
    lines.append("\n## Likelihood tables\n")
    for feature, table in likelihood_tables_dict.items():
        lines.append(f"### {feature}\n")
        lines.append(table.to_markdown(index=False))
        lines.append("")

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "outputs_task2_3")
    os.makedirs(output_dir, exist_ok=True)

    df = load_play_tennis_dataframe()
    features = ["Outlook", "Humidity", "Wind"]
    target = "Play"

    # Task 2 tables
    freq = frequency_tables(df, features, target)
    likelihood = likelihood_tables(df, features, target)
    save_tables_to_markdown(freq, likelihood, os.path.join(output_dir, "task2_tables.md"))

    # Task 2 condition from guide:
    condition_task2 = {"Outlook": "Rain", "Humidity": "High", "Wind": "Weak"}
    posterior_task2 = naive_bayes_posterior_manual(df, features, target, condition_task2, laplace_alpha=0.0)

    # Task 3 condition for variant 4:
    # Outlook=Sunny, Humidity=Normal, Wind=Strong
    condition_variant4 = {"Outlook": "Sunny", "Humidity": "Normal", "Wind": "Strong"}
    posterior_variant4 = naive_bayes_posterior_manual(df, features, target, condition_variant4, laplace_alpha=0.0)

    # Also verify using sklearn CategoricalNB
    sk_task2, sk_pred2 = fit_sklearn_categorical_nb(df, features, target, condition_task2)
    sk_v4, sk_pred4 = fit_sklearn_categorical_nb(df, features, target, condition_variant4)

    print("\n" + "#" * 78)
    print("Lab6 Task 2 - Bayes prediction for Outlook=Rain, Humidity=High, Wind=Weak")
    print("#" * 78)
    print("Manual posterior:", {k: round(v, 6) for k, v in posterior_task2.items()})
    print("Manual predicted class:", max(posterior_task2, key=posterior_task2.get))
    print("sklearn posterior:", {k: round(v, 6) for k, v in sk_task2.items()})
    print("sklearn predicted class:", sk_pred2)

    print("\n" + "#" * 78)
    print("Lab6 Task 3 (Variant 4) - Outlook=Sunny, Humidity=Normal, Wind=Strong")
    print("#" * 78)
    print("Manual posterior:", {k: round(v, 6) for k, v in posterior_variant4.items()})
    print("Manual predicted class:", max(posterior_variant4, key=posterior_variant4.get))
    print("sklearn posterior:", {k: round(v, 6) for k, v in sk_v4.items()})
    print("sklearn predicted class:", sk_pred4)
    print(f"\nSaved tables to: {os.path.join(output_dir, 'task2_tables.md')}")


if __name__ == "__main__":
    main()


import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "outputs_task4")
    os.makedirs(out_dir, exist_ok=True)

    dataset = fetch_california_housing()
    X, y = dataset.data, dataset.target
    feature_names = np.array(dataset.feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7
    )

    base_tree = DecisionTreeRegressor(max_depth=4, random_state=7)
    regressor = AdaBoostRegressor(
        estimator=base_tree,
        n_estimators=500,
        learning_rate=0.05,
        random_state=7,
    )
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    importances = regressor.feature_importances_
    importances = 100.0 * importances / np.max(importances)
    order = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(importances)), importances[order], color="tab:blue")
    plt.xticks(range(len(importances)), feature_names[order], rotation=45, ha="right")
    plt.ylabel("Relative importance (%)")
    plt.title("Task 2.4 - Relative feature importance (AdaBoostRegressor)")
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "feature_importance_adaboost.png")
    plt.savefig(fig_path, dpi=180)
    plt.close()

    print("\n" + "#" * 70)
    print("Task 2.4 - Relative feature importance")
    print("#" * 70)
    print(f"MAE  = {mae:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"R2   = {r2:.4f}")
    print("\nSorted feature importance:")
    for idx in order:
        print(f"{feature_names[idx]:>10}: {importances[idx]:7.3f}%")
    print(f"\nSaved chart: {fig_path}")


if __name__ == "__main__":
    main()


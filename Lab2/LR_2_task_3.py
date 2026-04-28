import numpy as np
from matplotlib import pyplot as plt
from pandas import read_csv
from pandas.plotting import scatter_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def main():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
    dataset = read_csv(url, names=names)

    print("dataset.shape =", dataset.shape)
    print(dataset.head(5))
    print("\nClass distribution:\n", dataset.groupby("class").size())

    # Basic plots (optional - comment out if you don't need charts)
    dataset.plot(kind="box", subplots=True, layout=(2, 2), sharex=False, sharey=False)
    plt.suptitle("Boxplots for Iris features")
    plt.show()

    dataset.hist()
    plt.suptitle("Histograms for Iris features")
    plt.show()

    scatter_matrix(dataset)
    plt.suptitle("Scatter Matrix for Iris")
    plt.show()

    array = dataset.values
    X = array[:, 0:4].astype(float)
    y = array[:, 4]

    X_train, X_validation, y_train, y_validation = train_test_split(
        X, y, test_size=0.20, random_state=1, stratify=y
    )

    models = []
    models.append(("LR", LogisticRegression(solver="liblinear", multi_class="ovr")))
    models.append(("LDA", LinearDiscriminantAnalysis()))
    models.append(("KNN", KNeighborsClassifier()))
    models.append(("CART", DecisionTreeClassifier()))
    models.append(("NB", GaussianNB()))
    models.append(("SVM", SVC(gamma="auto")))

    results = []
    names_ = []
    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring="accuracy")
        results.append(cv_results)
        names_.append(name)
        print(f"{name}: {cv_results.mean():.4f} ({cv_results.std():.4f})")

    plt.figure(figsize=(8, 4))
    plt.boxplot(results, labels=names_)
    plt.title("Algorithm Comparison (Iris)")
    plt.ylabel("Accuracy")
    plt.show()

    # Train final model (SVM as in methodical example)
    model = SVC(gamma="auto")
    model.fit(X_train, y_train)
    predictions = model.predict(X_validation)

    print("\nHold-out accuracy:", accuracy_score(y_validation, predictions))
    print("\nConfusion matrix:\n", confusion_matrix(y_validation, predictions))
    print("\nClassification report:\n", classification_report(y_validation, predictions))

    # Predict a new sample
    X_new = np.array([[5.0, 2.9, 1.0, 0.2]])
    pred = model.predict(X_new)[0]
    print("\nPrediction for X_new:", pred)


if __name__ == "__main__":
    main()


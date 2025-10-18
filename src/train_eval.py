import json
from pathlib import Path
from typing import Dict

import joblib
from sklearn.metrics import accuracy_score, f1_score

from . import config
from .data_loader import load_dataset, stratified_split
from .models.knn_model import grid_search_knn
from .models.nb_model import build_nb_pipeline
from .models.sbert_classifier import SbertSklearnClassifier
from .visualize import plot_class_distribution, plot_confusion, plot_predictions_bar, ensure_dir


def evaluate(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def run_experiment(dataset_path: Path = config.DATASET_PATH):
    ensure_dir(config.OUTPUT_DIR)
    ensure_dir(config.ARTIFACTS_DIR)

    print(f"Loading dataset: {dataset_path}")
    df = load_dataset(dataset_path)

    dist_path = config.OUTPUT_DIR / "class_distribution.png"
    plot_class_distribution(df["klasa"].tolist(), dist_path)

    # train/test split
    train_df, test_df = stratified_split(df, test_size=0.2, random_state=config.RANDOM_STATE)
    x_train = train_df["opis_problema"].tolist()
    y_train = train_df["klasa"].tolist()
    x_test = test_df["opis_problema"].tolist()
    y_test = test_df["klasa"].tolist()
    all_labels = sorted(df["klasa"].unique().tolist())

    metrics = {}

    # 1) Naive Bayes (baseline)
    print("Training Naive Bayes (TF-IDF + MultinomialNB)...")
    nb_pipe = build_nb_pipeline()
    nb_pipe.fit(x_train, y_train)
    nb_pred = nb_pipe.predict(x_test)
    metrics["naive_bayes"] = evaluate(y_test, nb_pred)
    plot_confusion(y_test, nb_pred, all_labels, "NB - Confusion Matrix", config.OUTPUT_DIR / "nb_confusion.png")
    plot_predictions_bar(nb_pred, all_labels, "NB - Predikcije po klasama", config.OUTPUT_DIR / "nb_predictions_bar.png")
    joblib.dump(nb_pipe, config.ARTIFACTS_DIR / "nb_pipeline.joblib")

    # 2) KNN (TF-IDF + KNN with GridSearch over k)
    print("Training KNN with GridSearch over k...")
    knn_search = grid_search_knn(x_train, y_train, k_values=config.KNN_K_VALUES)
    knn_best = knn_search.best_estimator_
    knn_pred = knn_best.predict(x_test)
    metrics["knn"] = evaluate(y_test, knn_pred)
    metrics["knn"]["best_k"] = int(knn_search.best_params_.get("clf__n_neighbors", -1))
    plot_confusion(y_test, knn_pred, all_labels, "KNN - Confusion Matrix", config.OUTPUT_DIR / "knn_confusion.png")
    plot_predictions_bar(knn_pred, all_labels, "KNN - Predikcije po klasama", config.OUTPUT_DIR / "knn_predictions_bar.png")
    joblib.dump(knn_best, config.ARTIFACTS_DIR / "knn_pipeline.joblib")

    # 3) Sentence-BERT + classifier (LogReg by default)
    print(f"Training SBERT ({config.SBERT_MODEL_NAME}) + {config.SBERT_CLASSIFIER} ...")
    sbert_clf = SbertSklearnClassifier()
    sbert_clf.fit(x_train, y_train)
    sbert_pred = sbert_clf.predict(x_test)
    metrics["sbert"] = evaluate(y_test, sbert_pred)
    metrics["sbert"]["classifier"] = config.SBERT_CLASSIFIER
    plot_confusion(y_test, sbert_pred, all_labels, "SBERT - Confusion Matrix", config.OUTPUT_DIR / "sbert_confusion.png")
    plot_predictions_bar(sbert_pred, all_labels, "SBERT - Predikcije po klasama", config.OUTPUT_DIR / "sbert_predictions_bar.png")
    joblib.dump(sbert_clf, config.ARTIFACTS_DIR / "sbert_classifier.joblib")

    with open(config.OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n=== Rezultati (Test set) ===")
    for name, m in metrics.items():
        acc = m.get("accuracy")
        f1 = m.get("f1_macro")
        extra = {k: v for k, v in m.items() if k not in {"accuracy", "f1_macro"}}
        print(f"{name:>10s}: accuracy={acc:.3f}, f1_macro={f1:.3f} {extra}")


if __name__ == "__main__":
    run_experiment()

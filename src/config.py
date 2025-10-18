from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "dataset_700.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# SBERT model and classifier settings
SBERT_MODEL_NAME = "sentence-transformers/LaBSE"
SBERT_CLASSIFIER = "logreg"  # options: "logreg" or "rf"
RANDOM_STATE = 42

# KNN settings
KNN_K_VALUES = [1, 3, 5, 7, 9]
KNN_METRIC = "cosine"  # recommended for TF-IDF

# TF-IDF settings
TFIDF_MIN_DF = 1
TFIDF_MAX_DF = 0.95
TFIDF_NGRAM_RANGE = (1, 2)

# Plots
FIGSIZE = (10, 6)
DPI = 120

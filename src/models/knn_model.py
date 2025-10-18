from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from ..preprocessing import PreprocessTextTransformer
from .. import config


def build_knn_pipeline(k: int = 5) -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", PreprocessTextTransformer()),
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=False,
                    ngram_range=config.TFIDF_NGRAM_RANGE,
                    min_df=config.TFIDF_MIN_DF,
                    max_df=config.TFIDF_MAX_DF,
                ),
            ),
            (
                "clf", KNeighborsClassifier(n_neighbors=k, metric=config.KNN_METRIC),
            ),
        ]
    )


def grid_search_knn(x_train, y_train, k_values: List[int] = None) -> GridSearchCV:
    if k_values is None:
        k_values = config.KNN_K_VALUES

    pipe = build_knn_pipeline()

    param_grid = {
        "clf__n_neighbors": k_values,
    }
    search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=5,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_train, y_train)
    return search

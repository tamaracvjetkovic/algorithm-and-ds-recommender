from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from ..preprocessing import PreprocessTextTransformer
from .. import config


def build_nb_pipeline() -> Pipeline:
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
                "clf", MultinomialNB()
            ),
        ]
    )

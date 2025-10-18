import re
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

STOPWORDS = {
    "ja", "ti", "on", "ona", "ono", "mi", "vi", "oni", "one",
    "mene", "tebe", "njega", "nje", "nas", "vas", "njih",
    "meni", "tebi", "njemu", "njoj", "nama", "vama", "njima",
    "moj", "tvoj", "njegov", "njen", "naš", "vaš", "njihov",

    "sam", "si", "je", "smo", "ste", "su",
    "jesam", "jesi", "jeste", "jesu",
    "bio", "bila", "bilo", "bili", "bile",
    "biti", "bih", "bi",

    "moram", "mora", "moramo", "morate", "moraju",
    "treba", "trebam", "trebamo", "trebate", "trebaju",
    "želim", "želi", "želimo", "želite", "žele",
    "hoću", "hoće", "hoćemo", "hoćete", "hoće",

    "kako", "šta", "kada", "zašto", "koji", "koja", "koje",

    "a", "ali", "pa", "te", "ni", "niti",
    "da", "ako", "kad", "dok", "jer", "što",

    "kroz", "kod", "nad", "pod", "pred",
    "uz", "oko", "pre",

    "vrlo", "veoma", "mnogo", "malo", "dosta",
    "još", "već", "samo", "takođe", "tako",
    "ovako", "onako",

    "neki", "neka", "neko", "neki",

    "nisam", "nisi", "nije", "nismo", "niste", "nisu",
    "niko", "ništa", "nikad",

    "li", "taj", "ta", "to", "ovaj", "ova", "ovo",
    "o", "ko", "se",
}

_punct_re = re.compile(r"[\.,;:!\?\(\)\[\]\{\}\-_'\"`~\/=+<>@#\$%\^&\*|\\]")
_space_re = re.compile(r"\s+")
_digit_re = re.compile(r"\d+")


def clean_text(text: str, remove_stopwords: bool = True) -> str:
    if text is None:
        return ""

    s = str(text).lower()
    s = _digit_re.sub(" ", s)  # remove digits
    s = _punct_re.sub(" ", s)  # remove punctuation
    s = _space_re.sub(" ", s).strip()  # normalize whitespace

    if remove_stopwords:
        tokens = [tok for tok in s.split(" ") if tok and tok not in STOPWORDS]
    else:
        tokens = [tok for tok in s.split(" ") if tok]

    return " ".join(tokens)


def clean_text_tfidf(text: str) -> str:
    return clean_text(text, remove_stopwords=True)


def clean_text_sbert(text: str) -> str:
    return clean_text(text, remove_stopwords=False)


class PreprocessTextTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X: Iterable[str], y=None):
        return self

    def transform(self, X: Iterable[str]):
        if isinstance(X, pd.Series):
            X = X.fillna("").astype(str).tolist()
        elif isinstance(X, (list, tuple, np.ndarray)):
            X = ["" if x is None else str(x) for x in X]
        else:
            X = ["" if x is None else str(x) for x in list(X)]

        return [clean_text_tfidf(x) for x in X]


def fill_nulls(df: pd.DataFrame, text_col: str = "opis_problema", label_col: str = "klasa") -> pd.DataFrame:
    df = df.copy()
    if text_col in df.columns:
        df[text_col] = df[text_col].fillna("")
    if label_col in df.columns:
        df = df.dropna(subset=[label_col])
    return df
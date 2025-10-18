from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from .preprocessing import fill_nulls


def load_dataset(path: Path, text_col: str = "opis_problema", label_col: str = "klasa") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = fill_nulls(df, text_col=text_col, label_col=label_col)
    # drop rows where text is empty after fill/cleaning
    df = df[df[text_col].astype(str).str.strip() != ""].reset_index(drop=True)
    return df


def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    text_col: str = "opis_problema",
    label_col: str = "klasa",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    y = df[label_col].values
    idx_train, idx_test = next(splitter.split(df, y))
    return df.iloc[idx_train].reset_index(drop=True), df.iloc[idx_test].reset_index(drop=True)

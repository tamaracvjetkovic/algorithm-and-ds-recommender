from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from .. import config
from ..preprocessing import clean_text_sbert


@dataclass
class SbertSklearnClassifier:
    model_name: str = config.SBERT_MODEL_NAME
    classifier_type: str = config.SBERT_CLASSIFIER
    random_state: int = config.RANDOM_STATE

    _sbert: Optional[SentenceTransformer] = None
    _clf: Optional[Any] = None
    _le: Optional[LabelEncoder] = None

    def _ensure_model(self):
        if self._sbert is None:
            self._sbert = SentenceTransformer(self.model_name)

    def _embed(self, texts: List[str]) -> np.ndarray:
        self._ensure_model()
        cleaned = [clean_text_sbert(t) for t in texts]
        # tokenizer + embedding
        embeddings = self._sbert.encode(cleaned, show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(embeddings)

    def fit(self, X: List[str], y: List[str]):
        self._le = LabelEncoder()
        y_enc = self._le.fit_transform(y)
        X_vec = self._embed(X)
        if self.classifier_type == "rf":
            self._clf = RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            self._clf = LogisticRegression(
                max_iter=2000,
                n_jobs=-1,
                class_weight=None,
                random_state=self.random_state,
                solver="lbfgs",
                multi_class="auto",
            )
        self._clf.fit(X_vec, y_enc)
        return self

    def predict(self, X: List[str]) -> np.ndarray:
        X_vec = self._embed(X)
        y_enc = self._clf.predict(X_vec)
        return self._le.inverse_transform(y_enc)

    def predict_proba(self, X: List[str]) -> np.ndarray:
        X_vec = self._embed(X)

        # classifier has predict_proba (LogisticRegression)
        if hasattr(self._clf, "predict_proba"):
            return self._clf.predict_proba(X_vec)  # ← Poziva sklearn-ov predict_proba

        # classifier has decision_function
        elif hasattr(self._clf, "decision_function"):
            scores = self._clf.decision_function(X_vec)
            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        else:
            raise AttributeError("Classifier does not support probability estimates")

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_sbert"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]):
        self.__dict__.update(state)
        if self._sbert is None:
            self._sbert = None

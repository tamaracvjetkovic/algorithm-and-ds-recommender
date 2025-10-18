from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from . import config


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def plot_class_distribution(labels: List[str], save_path: Path):
    ensure_dir(save_path.parent)
    plt.figure(figsize=config.FIGSIZE, dpi=config.DPI)
    sns.countplot(x=labels, order=sorted(set(labels)))
    plt.xticks(rotation=45, ha="right")
    plt.title("Distribucija klasa")
    plt.xlabel("Klasa")
    plt.ylabel("Broj primera")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion(y_true: List[str], y_pred: List[str], labels: List[str], title: str, save_path: Path):
    ensure_dir(save_path.parent)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=config.FIGSIZE, dpi=config.DPI)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, values_format="d")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_predictions_bar(y_pred: List[str], labels: List[str], title: str, save_path: Path):
    ensure_dir(save_path.parent)
    plt.figure(figsize=config.FIGSIZE, dpi=config.DPI)
    sns.countplot(x=y_pred, order=labels)
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.xlabel("Predikcija (klasa)")
    plt.ylabel("Broj predikcija")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

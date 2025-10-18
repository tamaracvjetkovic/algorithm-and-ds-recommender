import argparse

import joblib

from . import config


#python -m src.cli --text "your text here"

def load_model(name: str):
    name = name.lower()
    if name == "nb":
        path = config.ARTIFACTS_DIR / "nb_pipeline.joblib"
    elif name == "knn":
        path = config.ARTIFACTS_DIR / "knn_pipeline.joblib"
    elif name in {"sbert", "bert", "sentence-bert"}:
        path = config.ARTIFACTS_DIR / "sbert_classifier.joblib"
    else:
        raise ValueError("Nepoznat model. Koristite: nb, knn, sbert")

    if not path.exists():
        raise FileNotFoundError(f"Model nije pronađen: {path}. Pokrenite trening: python -m src.train_eval")

    model = joblib.load(path)
    return model


def main():
    parser = argparse.ArgumentParser(description="Sistem za preporuku algoritama/struktura podataka")
    parser.add_argument("--model", type=str, default="sbert", help="nb | knn | sbert")
    parser.add_argument("--text", type=str, required=True, help="Tekstualni opis problema")
    args = parser.parse_args()

    model = load_model(args.model)

    text = args.text
    pred = model.predict([text])[0]
    print(f"\nUlaz: {text}\nPreporuka: {pred}\n")

    try:
        probas = model.predict_proba([text])

        # top 3 preporuke
        classes = model._le.classes_
        top3_idx = probas[0].argsort()[-3:][::-1]
        for idx in top3_idx:
            print(f"{classes[idx]}: {probas[0][idx]*100:.1f}%")
    except:
        pass

if __name__ == "__main__":
    main()

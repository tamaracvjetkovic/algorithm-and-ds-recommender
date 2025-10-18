# Sistem za preporuku algoritama i struktura podataka

Tamara Cvjetkovic, SV48-2022

---

Uputstvo:

- Za Mac/Linux:
`python3 -m venv .venv`

- Za Windows:
`python -m venv .venv ..venv\Scripts\Activate`

`source .venv/bin/activates`

`pip install -r requirements.txt`

- iz root foldera projekta (sa aktivnim virtualnim okruženjem)
`python main.py`

---

Preporuke:
- SBERT (podrazumijevano)
`python -m src.cli --text "Želim da pronađem najkraći put između dva grada"`

- Naivni Bajes
`python -m src.cli --model nb --text "Moram da održavam najveći element nakon dodavanja i brisanja"`

- KNN
`python -m src.cli --model knn --text "Treba mi struktura za brzu pretragu po ključu"`

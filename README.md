# Fake News Classifier using BERT

Ce projet met en œuvre un modèle de classification binaire basé sur BERT pour détecter automatiquement des titres d'articles de presse fake vs vrais, en s'appuyant uniquement sur le champ `title`. Il utilise le modèle `bert-base-uncased` via la bibliothèque Hugging Face Transformers, avec une tête de classification personnalisée.

---

## Contenu du projet

- `main.ipynb` : pipeline complet (prétraitement, visualisation, entraînement, évaluation)
- `requirements.txt` : dépendances Python
- `.gitignore` : fichiers exclus du suivi Git

---

## Données

Les données proviennent du dataset [Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset) :

- `Fake.csv` : faux articles d’actualité
- `True.csv` : articles vérifiés
- Seule la colonne `title` est utilisée comme entrée du modèle.

---

## Visualisation exploratoire (Data Viz)

Une analyse exploratoire a été réalisée avant l’entraînement :

- Histogramme du nombre de mots par titre
- Répartition des classes (`Fake` vs `True`)
- Nuages de mots (wordclouds) pour chaque classe
- Projection t-SNE des mots les plus discriminants, basés sur les scores TF-IDF
- Visualisation de la sémantique BERT des mots dans l’espace vectoriel

Ces visualisations permettent de mieux comprendre les motifs lexicaux présents dans les fake news, et d’informer le choix de `max_length`.

---

## Prétraitement

- Création d’un label binaire : `1 = Fake`, `0 = True`
- Fusion des deux fichiers et stratification sur le label
- Tokenisation avec `BertTokenizerFast` (`bert-base-uncased`)
- Longueur maximale des séquences : `max_length = 15`
- Padding et troncation à longueur fixe

---

## Architecture du modèle

- **Modèle BERT** gelé (`bert-base-uncased`)
- **Tête de classification** :
  - Linear(768 → 512) → ReLU → Dropout
  - Linear(512 → 2) → LogSoftmax

---

## Entraînement

- Optimiseur : `AdamW`, `lr = 1e-5`
- Perte : `NLLLoss`
- Batch size : 32
- Epochs : 5
- Enregistrement du meilleur modèle (validation loss)
- Pas de fine-tuning de BERT (gel complet)

---

## Résultats (test)

| Classe     | Précision | Rappel | F1-score |
|------------|-----------|--------|----------|
| 0 (True)   | 0.80      | 0.89   | 0.84     |
| 1 (Fake)   | 0.89      | 0.79   | 0.84     |

- F1-score global : 0.84
- Courbes de perte visibles dans le notebook
- Matrice de confusion et `classification_report` affichés sur le test set

---

## Lancer le projet

```bash
git clone https://github.com/mounirfrhn/Fake-news-classifier.git
cd Fake-news-classifier

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook main.ipynb

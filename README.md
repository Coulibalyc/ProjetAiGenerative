# AI Job Recommender  
### Analyse sémantique des compétences & recommandation de métiers  
**SBERT · Gemini API · Streamlit**

---

## Description du projet

Ce projet vise à concevoir une **solution d’IA générative et de NLP sémantique** capable de :

- analyser un profil professionnel (compétences, projets, missions, outils) ;
- calculer la similarité avec un référentiel de métiers ;
- recommander le **métier le plus pertinent** ;
- fournir une **explication contextualisée** grâce à une API de GenAI (Gemini).

L’application est développée avec **Streamlit** afin d’offrir une interface web interactive et intuitive.

---

## Objectifs pédagogiques

Ce projet permet de :

- appliquer le **prétraitement de texte** et les **embeddings sémantiques** ;
- implémenter un moteur de **similarité sémantique avec SBERT** ;
- distinguer **scores numériques** et **analyse sémantique contextualisée** ;
- structurer un **référentiel de compétences professionnelles** ;
- intégrer une **API d’IA générative** de manière responsable et économique ;
- développer une **application web interactive** ;
- concevoir un **pipeline NLP complet** :
  - Nettoyage  
  - Embedding  
  - Scoring  
  - Recommandation  
  - Explication par IA générative  

---

## Architecture du projet

```bash
projet_ia_generative/
│
├── app.py
├── projet_ia_generative.ipynb
├── referentiel.csv
├── referentiel_embeddings.pkl
├── requirements.txt
│
├── .streamlit/
│   └── secrets.toml
│
└── README.md
```

---

## Technologies utilisées

### NLP & IA
- Sentence Transformers (SBERT) — `all-MiniLM-L6-v2`
- Google Gemini API

### Data & Calcul
- pandas, numpy
- scikit-learn (cosine similarity)

### Visualisation & Web
- Streamlit
- Plotly

---

## Installation

```bash
git clone <url-du-repo>
cd projet_ia_generative
python -m venv venv
```

Activation :

```bash
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate    # Windows
```

Installation des dépendances :

```bash
pip install -r requirements.txt
```

---

## Configuration Gemini

Créer `.streamlit/secrets.toml` :

```toml
GEMINI_API_KEY = "VOTRE_CLE_API"
```

---

## Lancer l’application

```bash
streamlit run app.py
```

---

##  Perspectives

- Pondération avancée
- Authentification
- Historique
- Déploiement cloud
- Multilingue

---

## Contexte académique

EFREI – Data Engineering & Artificial Intelligence  
Année : 2025–2026

---

## Licence

Projet pédagogique – usage libre pour l’apprentissage.

# 🔍 AI Job Recommender  
**Analyse sémantique des compétences & recommandation de métiers (SBERT + GenAI)**

---

## 📌 Description du projet

Ce projet a pour objectif de concevoir une **application d’intelligence artificielle** capable de recommander un métier à un utilisateur à partir de la description de :

- ses compétences,
- ses projets réalisés,
- ses missions professionnelles,
- ses outils maîtrisés.

L’approche repose sur :
- des **embeddings sémantiques (SBERT)** pour mesurer la similarité,
- un **référentiel structuré de compétences métiers**,
- l’intégration d’une **API de GenAI (Gemini)** pour enrichir et expliquer les résultats.

---

## 🎯 Objectifs pédagogiques

Ce projet permet de :

- appliquer le **prétraitement de texte** et les **embeddings sémantiques** ;
- distinguer **analyse numérique** (scores de similarité) et **analyse sémantique contextualisée** ;
- implémenter un **moteur de similarité basé sur SBERT** ;
- structurer un **référentiel de compétences professionnel** ;
- développer une **interface web interactive avec Streamlit** ;
- intégrer une **API de GenAI de manière responsable et contrôlée** ;
- concevoir un **pipeline NLP complet** (Nettoyage → Embeddings → Similarité → Recommandation → Explication IA).

---

## 🧠 Technologies utilisées

| Outil | Rôle |
|------|------|
| **Python** | Langage principal |
| **Streamlit** | Interface web |
| **Sentence-Transformers (SBERT)** | Embeddings sémantiques |
| **Scikit-learn** | Similarité cosinus |
| **Google Gemini API** | Reformulation & explication IA |
| **Pandas / NumPy** | Manipulation des données |
| **Plotly** | Visualisations interactives |

---

## ⚙️ Installation

### 1️⃣ Créer un environnement virtuel
```bash
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows


pip install -r requirements.txt

streamlit run app.py

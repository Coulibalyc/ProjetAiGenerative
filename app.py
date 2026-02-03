import streamlit as st
import pandas as pd
import numpy as np
import re
import logging
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from google import genai
import altair as alt
   
import plotly.express as px


st.set_page_config(page_title="AI Job Recommender", layout="wide")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Vérification de la clé API
if "GEMINI_API_KEY" not in st.secrets:
    st.error("Aucune clé API Gemini trouvée dans secrets.toml")
    st.stop()

# Gemini client
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def reformulate_text(text):
    try:
        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=f"Rewrite professionally in English without changing meaning: {text}"
        )
        return response.text
    except Exception as e:
        st.warning(f"Reformulation failed: {e}")
        return text


def explain_prediction(text, best_job):
    try:
        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=f"""
User profile:
{text}

Recommended job: {best_job}

Explain in 5 bullet points why this job matches.
"""
        )
        return response.text
    except Exception as e:
        return f"Explanation unavailable: {e}"



@st.cache_resource
def load_sbert():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


@st.cache_data
def load_referential():
    return pd.read_pickle("referentiel_embeddings.pkl")


model = load_sbert()
df = load_referential()
skill_matrix = np.vstack(df["embedding"].values)



st.title("Recommandateur d’emplois basé sur l’IA (SBERT + Gemini)")
st.write("Décrivez vos expériences professionnelles :")

user_skills = st.text_area("Compétences", height=120)
user_projects = st.text_area("Projets réalisés", height=120)
user_missions = st.text_area("Missions effectuées", height=120)
user_tools = st.text_area("Outils maîtrisés", height=120)

if st.button("Analyser"):

    # Fusion du texte
    user_full_text = " ".join([user_skills, user_projects, user_missions, user_tools]).strip()

    if len(user_full_text) == 0:
        st.warning("Veuillez remplir au moins un champ.")
        st.stop()

    # Étape 1 : nettoyage
    user_clean = clean_text(user_full_text)

    # 🔹 Étape 2 : reformulation GenAI
    st.info(" Reformulation du texte via Gemini…")
    user_ai_clean = reformulate_text(user_clean)

    # Étape 3 : SBERT
    user_emb = model.encode(user_ai_clean, device="cpu")

    # 🔹 Étape 4 : similarité
    scores = cosine_similarity([user_emb], skill_matrix)[0]
    df["similarity"] = scores

    job_scores = df.groupby("job")["similarity"].mean().sort_values(ascending=False)
    best_job = job_scores.index[0]

    # Étape 5 : explication
    explanation = explain_prediction(user_ai_clean, best_job)



    tab1, tab2, tab3 = st.tabs(["Résultat", "Graphiques", "Détails"])


    with tab1:
        st.success(f"Métier recommandé : **{best_job}**")
        st.write(explanation)

        st.subheader("Classement des métiers")
        st.dataframe(job_scores)

        st.subheader("Compétences les plus proches")
        st.dataframe(df.sort_values("similarity", ascending=False).head(10))


   
    with tab2:
        st.subheader("Similarité par métier")

        fig = px.bar(
            job_scores,
            orientation="h",
            title="Scores de similarité par métier",
            labels={"value": "Score", "index": "Métier"},
            color=job_scores.values,
            color_continuous_scale="Blues"
        )

        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)



    with tab3:
        st.write("Texte reformulé par Gemini :")
        st.info(user_ai_clean)

        st.write("Données complètes :")
        st.dataframe(df.sort_values("similarity", ascending=False))

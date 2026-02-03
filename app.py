import streamlit as st
import pandas as pd
import numpy as np
import re
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from google import genai
import plotly.express as px
import plotly.graph_objects as go
 

st.set_page_config(page_title="AI Job Recommender", page_icon="🧠", layout="wide")
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
 
# ---------- STYLE (CSS) ----------
CUSTOM_CSS = """
<style>
/* Page padding */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
 
/* Title */
h1, h2, h3 { letter-spacing: -0.5px; }
 
/* Cards */
.kpi-card {
  background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(16,185,129,0.12));
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 18px;
  padding: 16px 18px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.15);
}
.kpi-value { font-size: 28px; font-weight: 800; margin: 0; }
.kpi-label { opacity: 0.8; margin: 0; font-size: 13px; }
 
.job-card {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 14px 16px;
  box-shadow: 0 8px 22px rgba(0,0,0,0.18);
  height: 100%;
}
.job-title { font-size: 16px; font-weight: 800; margin-bottom: 6px; }
.job-score { font-size: 26px; font-weight: 900; margin: 0; }
.muted { opacity: .75; font-size: 12px; }
 
/* Buttons */
.stButton>button {
  border-radius: 14px !important;
  padding: 0.6rem 1rem !important;
  font-weight: 700 !important;
}
 
/* Sidebar */
section[data-testid="stSidebar"]{
  border-right: 1px solid rgba(255,255,255,0.08);
}
 
/* Tabs */
.stTabs [data-baseweb="tab"]{
  font-weight: 700;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
 

if "GEMINI_API_KEY" not in st.secrets:
    st.error("❌ Aucune clé API Gemini trouvée dans secrets.toml")
    st.stop()
 
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
 
def clean_text(text: str) -> str:
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
 

with st.sidebar:
    st.markdown("Paramètres")
    top_k = st.slider("Nombre de métiers à afficher", 3, 20, 10)
    show_details = st.toggle("Afficher les détails complets (table)", value=True)
    st.markdown("---")
    st.markdown("###Conseils de saisie")
    st.caption("• Écris en phrases complètes si possible\n\n• Ajoute des outils/technos\n\n• Ajoute le type de projets (API, dashboard, ETL, etc.)")
 

colA, colB = st.columns([3, 1], gap="large")
with colA:
    st.markdown("#AI Job Recommender")
    st.markdown(
        "Un recommandateur de métiers basé sur **SBERT (similarité sémantique)** "
        "+ **Gemini (reformulation & explication)**."
    )
 
st.markdown("---")
 

st.markdown("## Décris ton profil")
 
c1, c2 = st.columns(2, gap="large")
with c1:
    user_skills = st.text_area("Compétences", height=140, placeholder="Ex: SQL, Python, data viz, stats, ETL...")
    user_projects = st.text_area("Projets réalisés", height=140, placeholder="Ex: dashboard BI, RAG, web app, pipeline...")
with c2:
    user_missions = st.text_area("Missions effectuées", height=140, placeholder="Ex: nettoyage données, modèles, reporting...")
    user_tools = st.text_area("Outils maîtrisés", height=140, placeholder="Ex: PowerBI, Excel, Git, Docker, AWS...")
 
btn1, btn2, btn3 = st.columns([1, 1, 2])
with btn1:
    run = st.button("Analyser", use_container_width=True)
with btn2:
    clear = st.button("Réinitialiser", use_container_width=True)
with btn3:
    st.caption("💬 Plus ta description est précise, plus la recommandation est pertinente.")
 
if clear:
    st.session_state.clear()
    st.rerun()
 

if run:
    user_full_text = " ".join([user_skills, user_projects, user_missions, user_tools]).strip()
    if not user_full_text:
        st.warning("Veuillez remplir au moins un champ.")
        st.stop()
 
    user_clean = clean_text(user_full_text)
 
    with st.status("Traitement en cours…", expanded=True) as status:
        st.write("Nettoyage du texte")
        st.write("Reformulation via Gemini")
        user_ai_clean = reformulate_text(user_clean)
 
        st.write("Encodage SBERT")
        user_emb = model.encode(user_ai_clean, device="cpu")
 
        st.write("Calcul des similarités")
        scores = cosine_similarity([user_emb], skill_matrix)[0]
        df["similarity"] = scores
 
        st.write("Agrégation des scores par métier")
        job_scores = df.groupby("job")["similarity"].mean().sort_values(ascending=False)
 
        best_job = job_scores.index[0]
        best_score = float(job_scores.iloc[0])
 
        st.write("Génération de l’explication")
        explanation = explain_prediction(user_ai_clean, best_job)
 
        status.update(label="Terminé ", state="complete", expanded=False)
 
    st.markdown("---")
 
    k1, k2, k3, k4 = st.columns(4, gap="large")
    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <p class="kpi-label">Métier recommandé</p>
            <p class="kpi-value">{best_job}</p>
            <p class="muted">Top 1</p>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <p class="kpi-label">Score top 1</p>
            <p class="kpi-value">{best_score:.3f}</p>
            <p class="muted">Similarité moyenne</p>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <p class="kpi-label">Métiers évalués</p>
            <p class="kpi-value">{job_scores.shape[0]}</p>
            <p class="muted">Dans ton référentiel</p>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        top_gap = float(job_scores.iloc[0] - job_scores.iloc[1]) if job_scores.shape[0] > 1 else 0.0
        st.markdown(f"""
        <div class="kpi-card">
            <p class="kpi-label">Écart top 1 / top 2</p>
            <p class="kpi-value">{top_gap:.3f}</p>
            <p class="muted">Plus c’est grand, plus c’est net</p>
        </div>
        """, unsafe_allow_html=True)
 
    st.markdown("## 🏆 Résultats")
 
    top3 = job_scores.head(3)
    cards = st.columns(3, gap="large")
    for i, (job, score) in enumerate(top3.items()):
        with cards[i]:
            medal = "🥇" if i == 0 else ("🥈" if i == 1 else "🥉")
            st.markdown(
                f"""
                <div class="job-card">
                    <div class="job-title">{medal} {job}</div>
                    <p class="job-score">{float(score):.3f}</p>
                    <div class="muted">Score de similarité</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
 
    tab1, tab2, tab3 = st.tabs(["📌 Explication", "📊 Graphiques", "🔎 Détails"])
 
    with tab1:
        st.subheader("Pourquoi ce métier ?")
        st.write(explanation)
        st.info("Texte reformulé (utilisé pour la recherche) :")
        st.write(user_ai_clean)
 
    with tab2:
        st.subheader("Top métiers (barres)")
        topN = job_scores.head(top_k).sort_values(ascending=True)
 
        fig = px.bar(
            topN,
            orientation="h",
            title=f"Top {top_k} métiers (score de similarité)",
            labels={"value": "Score", "index": "Métier"},
        )
        fig.update_layout(height=550, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, use_container_width=True)
 
        st.subheader("Distribution des scores")
        dist = job_scores.values
        fig2 = go.Figure(data=go.Histogram(x=dist, nbinsx=18))
        fig2.update_layout(
            title="Distribution des similarités (tous métiers)",
            xaxis_title="Score",
            yaxis_title="Nombre de métiers",
            height=420,
            margin=dict(l=10, r=10, t=60, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)
 
    with tab3:
        st.subheader("Classement")
        st.dataframe(job_scores.head(top_k))
 
        st.subheader("Compétences/entrées les plus proches")
        st.dataframe(df.sort_values("similarity", ascending=False).head(10))
 
        if show_details:
            st.subheader("Table complète (triée par similarité)")
            st.dataframe(df.sort_values("similarity", ascending=False))
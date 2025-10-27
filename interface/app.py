import streamlit as st
import sys
import os
import re

# S'assurer que le dossier parent est dans le PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag_system.rag_pipeline import rag_analyze

# Configuration générale
st.set_page_config(
    page_title="Fake News Detector",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Titre de la page
st.title("Fake News Detection System")
st.write("Analysez une affirmation et obtenez une évaluation automatique de sa véracité.")

# Stocker l’historique dans la session
if "history" not in st.session_state:
    st.session_state.history = []

# Zone de saisie utilisateur
user_input = st.text_area("Entrez le texte à analyser :", height=120, placeholder="Exemple : Donald Trump is not the president of the United States")

# Bouton principal
if st.button("Analyser"):
    if user_input.strip():
        with st.spinner("Analyse en cours..."):
            result = rag_analyze(user_input)

        result_text = result.get("result_text", "")

        # Extraction du verdict et de l'explication
        if isinstance(result_text, dict):
            verdict = result_text.get("verdict", "").upper()
            explanation = result_text.get("explanation", "")
        else:
            verdict_match = re.search(r"Verdict:\s*(\w+)", result_text, re.IGNORECASE)
            explanation_match = re.search(r"Explanation:\s*(.*)", result_text, re.IGNORECASE)
            verdict = verdict_match.group(1).upper() if verdict_match else "FAKE"
            explanation = explanation_match.group(1).strip() if explanation_match else ""

        # Normalisation du verdict
        if verdict not in ["TRUE", "FAKE"]:
            verdict = "FAKE"

        # Enregistrement dans l’historique
        st.session_state.history.insert(0, {
            "text": user_input,
            "verdict": verdict,
            "explanation": explanation,
            "duration": f"{result.get('eval_duration', 0):.2f}s"
        })

# Affichage du dernier résultat
if st.session_state.history:
    latest = st.session_state.history[0]
    st.subheader("Résultat actuel")
    st.markdown(f"**Verdict :** {latest['verdict']}")
    st.markdown(f"**Explication :** {latest['explanation']}")
    st.caption(f"Durée d'analyse : {latest['duration']}")

# Historique des analyses précédentes
if len(st.session_state.history) > 1:
    st.markdown("---")
    st.subheader("Historique des analyses")
    for i, entry in enumerate(st.session_state.history[1:], start=1):
        with st.expander(f"Analyse #{i} — Verdict : {entry['verdict']}"):
            st.write(f"**Texte analysé :** {entry['text']}")
            st.write(f"**Explication :** {entry['explanation']}")
            st.caption(f"Durée : {entry['duration']}")

# Style CSS simple et élégant
st.markdown("""
<style>
    body {
        background-color: #f9f9f9;
        color: #333;
    }
    .stButton>button {
        background-color: #2e7bcf;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.6em 1.2em;
        font-weight: 500;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #155a9c;
        transform: scale(1.02);
    }
    .stTextArea textarea {
        border-radius: 6px;
        border: 1px solid #ccc;
        font-size: 15px;
    }
    .stExpander {
        border: 1px solid #ddd;
        border-radius: 6px;
        background-color: #f7f7f7;
    }
</style>
""", unsafe_allow_html=True)

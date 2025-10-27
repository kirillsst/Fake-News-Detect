# import streamlit as st
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
from rag_system.rag_pipeline import rag_analyze

st.title("Fake News Checker RAG")

user_text = st.text_area("Entrez le texte de l'actualité :")

if st.button("Vérifier"):
    if user_text.strip() == "":
        st.warning("Entrez le texte à analyser !")
    else:
        verdict = rag_analyze(user_text)
        st.subheader("Résultat RAG")
        st.write(verdict)

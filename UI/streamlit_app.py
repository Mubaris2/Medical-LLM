import streamlit as st
import requests

API_BASE = st.secrets.get("api_url", "http://localhost:8000")

st.set_page_config(page_title="Medical LLM Demo", layout="centered")

st.title("Medical-LLM — Demo")

mode = st.radio("Mode", ["Q&A (RAG)", "Disease Summary", "Symptom Checker"])

if mode == "Q&A (RAG)":
    q = st.text_area("Ask a medical question", height=120)
    top_k = st.slider("Top-k retrieved docs", 1, 10, 3)
    if st.button("Ask"):
        with st.spinner("Querying model..."):
            resp = requests.post(f"{API_BASE}/query", json={"query": q, "top_k": top_k}).json()
            st.subheader("Answer")
            st.write(resp["answer"])
            st.subheader("Top Documents")
            for d in resp["retrieved_docs"]:
                st.markdown("---")
                st.write(d[:800] + ("..." if len(d) > 800 else ""))

elif mode == "Disease Summary":
    disease = st.text_input("Disease name")
    if st.button("Get Summary"):
        r = requests.post(f"{API_BASE}/disease_summary", json={"disease": disease}).json()
        st.write(r["structured"])

else:
    symptoms = st.text_area("Enter symptoms (comma separated)")
    top_k = st.number_input("Top K suggestions", min_value=1, max_value=10, value=5)
    if st.button("Check"):
        s_list = [s.strip() for s in symptoms.split(",") if s.strip()]
        r = requests.post(f"{API_BASE}/symptom_checker", json={"symptoms": s_list, "top_k": top_k}).json()
        st.subheader("Predictions")
        for p in r:
            st.write(f"**{p['disease']}** — {p['reason']} — Confidence: {p['confidence']:.2f}")

import streamlit as st
import requests

st.title("AskNYC — NYC Zoning Assistant")
question = st.text_input("Ask about NYC zoning:")
if question:
    try:
        r = requests.post("http://localhost:8000/ask", json={"query": question})
        data = r.json()
        st.write(data["answer"])
        st.caption("Sources: " + ", ".join(data["sources"]))
    except requests.exceptions.ConnectionError:
        st.error("Backend not running — start it with: uvicorn main:app --reload")

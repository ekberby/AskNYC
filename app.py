import streamlit as st
from main import ask          # importing runs main's setup once (loads chunks, builds index, loads model)

st.title("AskNYC — NYC Zoning Assistant")
question = st.text_input("Ask about NYC zoning:")
if question:
    data = ask(question)                       # returns {"answer":..., "sources":[...]}
    st.write(data["answer"])
    st.caption("Sources: " + ", ".join(data["sources"]))
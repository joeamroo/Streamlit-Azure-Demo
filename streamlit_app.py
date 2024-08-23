import sys
import os
import logging
import streamlit as st
from rag_pipeline import rag_pipeline_run, initialize_document_stores, FastembedTextEmbedder

# Streamlit app UI
st.title("RAG Pipeline Demo")

query = st.text_input("Enter your query:")

if query:
    # Run the RAG pipeline
    answer, sources, images = rag_pipeline_run(query)

    st.write("Expert Answer:")
    st.write(answer)

    st.write("Sources:")
    for source in sources:
        st.write(f"- {source}")

    if images:
        st.write("Relevant Images:")
        for img in images:
            st.image(img)

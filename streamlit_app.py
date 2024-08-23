import os
import streamlit as st
from rag_pipeline import rag_pipeline_run, embedder  # Ensure proper import of embedder and generator

# Streamlit app UI
st.title("RAG Pipeline Demo with Embedding")

# Set environment variables for Azure Search and OpenAI API keys
os.environ["AZURE_SEARCH_API_KEY"] = st.secrets["AZURE_SEARCH_API_KEY"]
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

query = st.text_input("Enter your query:")

if query:
    # Run the RAG pipeline with the required arguments
    answer, sources, images = rag_pipeline_run(query, embedder)

    st.write("Expert Answer:")
    st.write(answer)

    st.write("Sources:")
    for source in sources:
        st.write(f"- {source}")

    if images:
        st.write("Relevant Images:")
        for img in images:
            st.image(img)

import os
import streamlit as st
from rag_pipeline import rag_pipeline_run, FastembedTextEmbedder

# Streamlit app UI
st.title("RAG Pipeline Demo with Embedding")

# Set up environment variables for Azure Search and OpenAI
os.environ["AZURE_SEARCH_API_KEY"] = st.secrets["AZURE_SEARCH_API_KEY"]
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Initialize the FastembedTextEmbedder
embedder = FastembedTextEmbedder(model="BAAI/bge-small-en-v1.5")
embedder.warm_up()

query = st.text_input("Enter your query:")

if query:
    # Run the RAG pipeline with the required arguments
    answer = rag_pipeline_run(query, embedder)

    st.write("Expert Answer:")
    st.write(answer)

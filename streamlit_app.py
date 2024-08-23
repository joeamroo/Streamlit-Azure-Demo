import streamlit as st
from rag_pipeline import rag_pipeline_run, initialize_document_stores, FastembedTextEmbedder

# Initialize Haystack components
document_stores = initialize_document_stores()
embedder = FastembedTextEmbedder(model="BAAI/bge-small-en-v1.5")
embedder.warm_up()

# Streamlit app UI
st.title("RAG Pipeline Demo")

query = st.text_input("Enter your query:")

if query:
    # Run the RAG pipeline with the required arguments
    answer, sources, images = rag_pipeline_run(query, document_stores, embedder)

    st.write("Expert Answer:")
    st.write(answer)

    st.write("Sources:")
    for source in sources:
        st.write(f"- {source}")

    if images:
        st.write("Relevant Images:")
        for img in images:
            st.image(img)

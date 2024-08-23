import streamlit as st
from rag_pipeline import rag_pipeline_run, initialize_document_stores

# Set up environment variables for Azure Search and OpenAI keys
os.environ["AZURE_SEARCH_API_KEY"] = st.secrets["AZURE_SEARCH_API_KEY"]
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Initialize the Azure Search retriever with semantic search
document_stores = initialize_document_stores()
embedder = FastembedTextEmbedder(model="BAAI/bge-small-en-v1.5")
embedder.warm_up()

st.title("RAG Pipeline with Azure Cognitive Search")
query = st.text_input("Enter your query:")

if query:
    response, sources, images = rag_pipeline_run(query, document_stores, embedder)
    
    st.write("Expert Answer:")
    st.write(response)
    
    st.write("Sources:")
    for source in sources:
        st.write(f"- [{source['document_type']}] {source['title']} (Chunk {source['chunk_index']}/{source['total_chunks']}) from {source['collection']} collection")

    if images:
        st.write("Relevant Images:")
        for img in images:
            st.image(img)

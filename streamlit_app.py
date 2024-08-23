import sys
import os
import logging
import streamlit as st
from rag_pipeline import rag_pipeline_run, initialize_document_stores, FastembedTextEmbedder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the home directory to the Python path (if needed)
sys.path.append(os.path.expanduser('~'))

# Initialize Haystack components
try:
    document_stores = initialize_document_stores()
    embedder = FastembedTextEmbedder(model="BAAI/bge-small-en-v1.5")
    embedder.warm_up()
except Exception as e:
    logger.error(f"Error initializing components: {str(e)}")
    st.error(f"Failed to initialize components: {str(e)}")

# Streamlit app UI
st.title("RAG Pipeline Demo")
query = st.text_input("Enter your query:")

if query:
    try:
        # Run the RAG pipeline
        response, sources, images = rag_pipeline_run(query, document_stores, embedder)

        # Display the response
        st.write("Expert Answer:")
        st.write(response)

        # Display the sources
        st.write("Sources:")
        for source in sources:
            st.write(f"- [{source['document_type']}] {source['title']} (Chunk {source['chunk_index']}/{source['total_chunks']}) from {source['collection']} collection")

        # Display the images
        if images:
            st.write("Relevant Images:")
            for img in images:
                st.image(img)
    except Exception as e:
        logger.error(f"Error running the RAG pipeline: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

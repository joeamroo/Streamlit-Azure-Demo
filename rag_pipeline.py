import os
import logging
from azure_search_retriever import AzureSearchRetriever
from haystack.components.generators import OpenAIGenerator
from haystack_integrations.components.embedders.fastembed import FastembedTextEmbedder

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Azure Search configuration
search_service_endpoint = "https://dwlouisaicognitive.search.windows.net"
index_name = "testindex"
api_key = os.environ["AZURE_SEARCH_API_KEY"]

# Initialize the Azure Search Retriever
retriever = AzureSearchRetriever(search_service_endpoint, index_name, api_key)

# Initialize the FastembedTextEmbedder
embedder = FastembedTextEmbedder(model="BAAI/bge-small-en-v1.5")

# Initialize the OpenAIGenerator with the API key directly
generator = OpenAIGenerator(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o-mini")

def rag_pipeline_run(query):
    # Retrieve documents using Azure Search
    retrieved_texts = retriever.retrieve(query)
    
    # Generate embeddings for the query
    embedding_result = embedder.run(text=query)
    query_embedding = embedding_result["embedding"]

    # Process the retrieved texts and generate the response
    formatted_documents = [text for text in retrieved_texts]
    
    # Generate a response using the OpenAI API
    response = generator.run(formatted_documents)

    return response, formatted_documents, []

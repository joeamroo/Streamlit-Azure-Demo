import os
import logging
from azure_search_retriever import AzureSearchRetriever
from haystack.components.generators import OpenAIGenerator

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Azure Search configuration
search_service_endpoint = "https://dwlouisaicognitive.search.windows.net"
index_name = "testindex"
api_key = os.environ["AZURE_SEARCH_API_KEY"]

# Initialize the Azure Search Retriever
retriever = AzureSearchRetriever(search_service_endpoint, index_name, api_key)

# Initialize the OpenAI Generator
generator = OpenAIGenerator(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o-mini")

def rag_pipeline_run(query, embedder):
    # Retrieve documents using Azure Search
    retrieved_texts = retriever.retrieve(query)
    
    # Combine retrieved texts into a single context
    context = "\n".join(retrieved_texts)

    # Generate embeddings for the query (if needed for further processing)
    embedding_result = embedder.run(text=query)
    query_embedding = embedding_result["embedding"]

    # Generate a response from OpenAI using the context
    prompt = f"Based on the following context, answer the query:\n\nContext:\n{context}\n\nQuery: {query}\n\nAnswer:"
    response = generator.run(prompt=prompt)
    
    # Return only the generated answer
    answer = response["replies"][0]
    return answer

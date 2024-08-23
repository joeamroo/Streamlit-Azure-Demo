import os
import logging
from azure_search_retriever import AzureSearchRetriever
from haystack.components.generators import OpenAIGenerator
from haystack_integrations.components.embedders.fastembed import FastembedTextEmbedder
from haystack.utils import Secret

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Azure Search configuration
search_service_endpoint = "https://dwlouisaicognitive.search.windows.net"
index_name = "testindex"
api_key = os.environ["AZURE_SEARCH_API_KEY"]

# Initialize the Azure Search Retriever
retriever = AzureSearchRetriever(search_service_endpoint, index_name, api_key)

# Initialize the FastembedTextEmbedder with progress_bar set to False
embedder = FastembedTextEmbedder(model="BAAI/bge-small-en-v1.5", progress_bar=False)

# Warm up the embedder
embedder.warm_up()

# Initialize the OpenAIGenerator
generator = OpenAIGenerator(api_key=Secret.from_env_var("OPENAI_API_KEY"), model="gpt-4o-mini")

def rag_pipeline_run(query):
    # Retrieve documents using Azure Search
    retrieved_texts = retriever.retrieve(query)
    
    # Generate embeddings for the query
    embedding_result = embedder.run(text=query)
    query_embedding = embedding_result["embedding"]

    # Format the query as a chat message for OpenAI
    chat_input = [{"role": "user", "content": query}]

    # Generate the response using the OpenAI generator
    response = generator.run(messages=chat_input)

    # Process the retrieved texts and generate the response
    formatted_documents = [text for text in retrieved_texts]

    return response["choices"][0]["message"]["content"], formatted_documents, []

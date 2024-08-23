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

# Initialize the FastembedTextEmbedder with tqdm disabled
embedder = FastembedTextEmbedder(model="BAAI/bge-small-en-v1.5")
embedder.warm_up()

# Initialize the OpenAIGenerator with the API key wrapped in Secret
openai_api_key = os.environ["OPENAI_API_KEY"]
generator = OpenAIGenerator(api_key=Secret.from_token(openai_api_key), model="gpt-4o-mini")

def rag_pipeline_run(query, embedder):
    # Retrieve documents using Azure Search
    retrieved_texts = retriever.retrieve(query)
    
    # Generate embeddings for the query (you can use this for other purposes)
    embedding_result = embedder.run(text=query)
    query_embedding = embedding_result["embedding"]

    # Generate a response using the OpenAI API with the original query
    response = generator.run([query])

    # Process the response to return just the text
    answer = response['choices'][0]['message']['content']

    return answer, retrieved_texts, []

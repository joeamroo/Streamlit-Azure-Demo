import os
import logging
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret
from haystack_integrations.components.embedders.fastembed import FastembedTextEmbedder
from azure_search_retriever import AzureSearchRetriever

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Azure Search configuration
search_service_endpoint = "https://dwlouisaicognitive.search.windows.net"
index_name = "testindex"
api_key = os.getenv("AZURE_SEARCH_API_KEY")

# Initialize the Azure Search Retriever
retriever = AzureSearchRetriever(search_service_endpoint, index_name, api_key)

# Initialize the FastembedTextEmbedder
embedder = FastembedTextEmbedder(model="BAAI/bge-small-en-v1.5")

# Warm up the embedder
embedder.warm_up()

# Initialize the OpenAIGenerator with the appropriate temperature
generator = OpenAIGenerator(
    api_key=Secret.from_env_var("OPENAI_API_KEY"),
    model="gpt-4o-mini",
    generation_kwargs={"temperature": 0.2}
)

def rag_pipeline_run(query):
    # Retrieve documents using Azure Search
    retrieved_texts = retriever.retrieve(query)
    
    # If no documents are retrieved, return a message indicating that
    if not retrieved_texts:
        return "No relevant documents found.", [], []

    # Construct the prompt for GPT-4
    prompt = f"""
    You are an expert in Profrac Procedures. Your task is to answer questions based on the provided documents.

    The documents contain technical content. Use the provided information to deliver a precise and accurate response.

    When discussing technical concepts, briefly explain them in a way that would be understandable to someone with general knowledge of the topic. Always mention the documents you sourced the answers from, and provide a link to them if possible.

    Retrieved Documents:
    {"\n".join(retrieved_texts)}

    Question: {query}
    Expert Answer:
    """

    # Debugging: Log the generated prompt
    logger.info(f"Generated prompt: {prompt}")

    # Generate the answer using GPT-4
    response = generator.run(prompt=prompt)

    # Consolidate the answer and meta information
    answer = response.get("replies", [])[0] if response.get("replies") else "No response generated."

    return answer, [], []

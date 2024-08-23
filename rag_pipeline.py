import os
import logging
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret
from haystack_integrations.components.embedders.fastembed import FastembedTextEmbedder
from azure_search_retriever import AzureSearchRetriever

# Initialize logging
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
    retrieved_documents = retriever.retrieve(query)
    
    # Collect and structure figures, images, and URLs
    figures_and_images = []
    sources = []
    for doc in retrieved_documents:
        for page in doc["content"]["pages"]:
            figures_and_images.extend(page.get("figures", []))
            for table in page.get("tables", []):
                figures_and_images.append(table["tbl_img_name"])

            sources.append({
                "document_type": "PDF",
                "title": page.get("document_name", "Unknown Document"),
                "link": page.get("url", "#")
            })

    # Generate embeddings for the query
    embedding_result = embedder.run(text=query)
    query_embedding = embedding_result["embedding"]
    
    # Construct the prompt for GPT-4
    prompt = f"""
    You are an expert in Profrac Procedures. Your task is to answer questions based on the provided documents, including any figures or images.

    The documents contain technical content with references to figures and images. These may include important diagrams, charts, or visual information, or tables related to the topic. Use these as necessary to provide a complete answer.

    When discussing technical concepts, briefly explain them in a way that would be understandable to someone with general knowledge of the topic. Always mention the documents you sourced the answers from, and provide a link to them from the metadata if possible.

    Retrieved Documents:
    {retrieved_documents}

    Figures and Images:
    {figures_and_images}

    Question: {query}
    Expert Answer:
    """

    # Debugging: Log the generated prompt
    logger.info(f"Generated prompt: {prompt}")

    # Generate the answer using GPT-4
    response = generator.run(prompt=prompt)

    # Consolidate the answer and meta information
    answer = response.get("replies", [])[0] if response.get("replies") else "No response generated."

    return answer, sources, figures_and_images

import logging
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret
from haystack_integrations.components.embedders.fastembed import FastembedTextEmbedder

# Initialize logging
logger = logging.getLogger(__name__)

# Initialize the FastembedTextEmbedder
embedder = FastembedTextEmbedder(model="BAAI/bge-small-en-v1.5", disable_tqdm=True, progress_bar=False)
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
    
    # Collect figures and images (assuming these are stored in the Azure index under specific fields)
    figures_and_images = [f"{doc.meta.get('figure_name', '')}: {doc.content}" for doc in retrieved_texts if 'figure' in doc.meta]
    
    # Generate embeddings for the query
    embedding_result = embedder.run(text=query)
    query_embedding = embedding_result["embedding"]
    
    # Construct the prompt for GPT-4
    prompt = f"""
    You are an expert in Profrac Procedures. Your task is to answer questions based on the provided documents, including any figures or images.

    The documents contain technical content with references to figures and images. These may include important diagrams, charts, or visual information, or tables related to the topic. Use these as necessary to provide a complete answer.

    When discussing technical concepts, briefly explain them in a way that would be understandable to someone with general knowledge of the topic. Always mention the documents you sourced the answers from, and provide a link to them from the metadata if possible.

    Retrieved Documents:
    {retrieved_texts}

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
    
    # Extracting sources or additional meta information if needed
    sources = [
        {
            "document_type": doc.meta.get('document_type', 'Unknown'),
            "title": doc.meta.get('title', 'Untitled'),
            "link": doc.meta.get('link', '#')
        }
        for doc in retrieved_texts
    ]

    # If there are images or figures, add them to the sources
    if figures_and_images:
        sources.append({"figure_or_image": figures_and_images})

    return answer, sources, figures_and_images

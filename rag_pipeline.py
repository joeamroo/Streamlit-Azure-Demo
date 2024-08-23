import os
import logging
import re
from typing import List, Dict, Tuple
from haystack import Pipeline
from haystack.dataclasses import Document
from haystack.components.generators import OpenAIGenerator
from haystack_integrations.components.embedders.fastembed import FastembedTextEmbedder
from haystack import component

from azure_search_retriever import AzureSearchRetriever

# Disable telemetry
os.environ["HAYSTACK_TELEMETRY_ENABLED"] = "False"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_document_stores():
    return {
        "azure_search": AzureSearchRetriever(
            search_service_endpoint="https://dwlouisaicognitive.search.windows.net",
            index_name="testindex",
            api_key=os.getenv("AZURE_SEARCH_API_KEY"),
            semantic_config_name="testsemanticconfig"  # Your semantic configuration name
        )
    }

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def truncate_prompt(prompt: str, max_tokens: int = 128000) -> str:
    current_tokens = count_tokens(prompt)
    if current_tokens > max_tokens:
        excess_tokens = current_tokens - max_tokens
        truncated_prompt = prompt[:-excess_tokens]
        logger.warning(f"Prompt truncated to fit within {max_tokens} tokens.")
        return truncated_prompt
    return prompt

@component
class CustomPromptTemplate:
    @component.output_types(prompt=str)
    def run(self, documents: List[Document], query: str):
        context = "\n".join([
            f"[{doc.meta.get('document_type', doc.meta.get('content_type', 'Unknown'))} - "
            f"{doc.meta.get('title', doc.meta.get('term', 'Untitled'))}] "
            f"(Chunk {doc.meta.get('chunk_index', 'N/A')}/{doc.meta.get('total_chunks', 'N/A')}) {doc.content}"
            for doc in documents
        ])
        prompt = f"""
        You are an expert in Oil & Gas and the Energy Industry. Your task is to answer questions based on the provided documents. Always remind the user of any safety concerns in an Answer if there's any.
        Answer the question primarily using the information from the retrieved documents. If the documents don't contain enough information to fully answer the question, you may use your expert knowledge to supplement the answer. Clearly indicate when you're using information beyond what's provided in the documents.
        For each piece of information you use from the documents, provide a citation using the following format: (Source:  - title/term) If source/title/term is "unknown" its "Podcast"!! return the date, author and publisher (Only return the pieces of information available, if a piece of information is unknown/none don't return it)
        If there are any image links in the content, describe them if they are relevant to answering the question. These images may contain important diagrams, charts, or visual information related to the topic.
        When discussing technical concepts, briefly explain them in a way that would be understandable to someone with a general knowledge of the topic.
        Retrieved Documents:
        {context}
        Question: {query}
        Expert Answer:
        """
        return {"prompt": prompt}

def rag_pipeline_run(
    query: str,
    document_stores: Dict[str, AzureSearchRetriever],
    embedder: FastembedTextEmbedder,
    debug: bool = False
) -> Tuple[str, List[Dict], List[str]]:
    logger.info(f"Running RAG pipeline for query: '{query}'")
    
    try:
        all_documents = []
        for collection, document_store in document_stores.items():
            try:
                result = document_store.run(query=query, top_k=10)
                all_documents.extend(result["documents"])
                logger.info(f"Retrieved {len(result['documents'])} documents from '{collection}' collection")
            except Exception as e:
                logger.error(f"Error retrieving documents from '{collection}': {str(e)}")
        
        if not all_documents:
            logger.warning("No documents were retrieved from any collection.")
            return "I'm sorry, but I couldn't find any relevant information to answer your query. Could you please rephrase or ask a different question?", [], []

        # Generate the prompt as before
        prompt_template = CustomPromptTemplate()
        prompt_result = prompt_template.run(documents=all_documents, query=query)
        prompt = prompt_result["prompt"]
        
        logger.info(f"Generated prompt: {prompt}")
        prompt = truncate_prompt(prompt, max_tokens=128000)

        # Call OpenAI API
        generator = OpenAIGenerator(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")
        response = generator.run(prompt=prompt)
        answer = response["replies"][0]

        # Extract sources and images
        sources = [
            {
                "document_type": doc.meta.get('document_type', doc.meta.get('content_type', 'Unknown')),
                "title": doc.meta.get('title', doc.meta.get('term', 'Untitled')),
                "chunk_index": doc.meta.get('chunk_index', 'N/A'),
                "total_chunks": doc.meta.get('total_chunks', 'N/A'),
                "collection": doc.meta.get('collection', 'Unknown'),
                "link": doc.meta.get('link', '#')
            }
            for doc in all_documents
        ]

        images = []
        for doc in all_documents:
            img_links = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', doc.content)
            images.extend([link for link in img_links if link.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))])

           # Debugging: Log the generated answer, sources, and images
            logger.info(f"Generated answer: {answer}")
            logger.info(f"Sources: {sources}")
            logger.info(f"Images: {images}")

            return answer, sources, images
       
    except Exception as e:
            logger.error(f"Error in RAG pipeline: {str(e)}")
            return f"An error occurred while processing your query: {str(e)}", [], []


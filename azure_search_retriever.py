from azure.search.documents import SearchClient, SearchOptions
from haystack import component
from haystack.dataclasses import Document

@component
class AzureSearchRetriever:
    def __init__(self, search_service_endpoint, index_name, api_key, semantic_config_name=None):
        self.client = SearchClient(
            endpoint=search_service_endpoint,
            index_name=index_name,
            credential=api_key
        )
        self.semantic_config_name = semantic_config_name

    def run(self, query: str, top_k: int = 10):
        search_options = SearchOptions(
            query_type="semantic",
            semantic_configuration_name=self.semantic_config_name,
            size=top_k,
            query_language="en-us",
            answers="extractive|count-3",
            captions="extractive"
        )
        
        results = self.client.search(query=query, search_options=search_options)
        documents = []

        for result in results:
            flattened_text = self.flatten_text(result['content'])
            doc = Document(content=flattened_text, meta=result)
            documents.append(doc)
        
        return {"documents": documents}

    def flatten_text(self, content):
        flattened_text = ""
        if "pages" in content:
            for page in content["pages"]:
                flattened_text += page.get("text", "") + " "
                if "figures" in page:
                    for figure in page["figures"]:
                        flattened_text += figure.get("ocr_text", "") + " "
                if "tables" in page:
                    flattened_text += table.get("content", "") + " "
        return flattened_text.strip()

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

class AzureSearchRetriever:
    def __init__(self, search_service_endpoint, index_name, api_key):
        self.search_client = SearchClient(
            endpoint=search_service_endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(api_key)
        )

    def retrieve(self, query):
        # Perform the search using the search text
        results = self.search_client.search(
            search_text=query,
            select=["content/pages/text"],  # Select only the relevant nested text field
            top=10,  # Adjust the number of results returned
            include_total_count=True  # To include the total count of results
        )
        
        # Flatten the text from the nested structure
        flattened_results = []
        for result in results:
            if 'content' in result and 'pages' in result['content']:
                for page in result['content']['pages']:
                    if 'text' in page:
                        flattened_results.append(page['text'])
        
        return flattened_results

import logging
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from geo_agent.search_engine import SearchManager
from geo_agent.retrieval import ContentLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

class WebSearchPipe:
    """
    Orchestrates search and retrieval.
    """
    def __init__(self, config_path='geo_agent/config.yaml'):
        self.search_manager = SearchManager(config_path)
        self.content_loader = ContentLoader(config_path)

    def search_and_retrieve(self, query: str):
        # 1. Search
        results = self.search_manager.search(query)
        doc_list = []
        
        # 2. Retrieve content for each result
        for result in results:
            print(f"Found result: {result.title} - {result.url}")
            if not result.url:
                continue

            content = self.content_loader.process(result)
            if content is not None:
                result.snippet = content
                result.idx = len(doc_list) + 1
                doc_list.append(result)
        
        return doc_list

if __name__ == "__main__":
    agent = WebSearchPipe()
    docs = agent.search_and_retrieve("mention the names of any 3 famous folklore sports in karnataka state")
    for doc in docs:
        print(f"URL: {doc.url}\nContent Preview:\n{doc.snippet[:500]}\n{'-'*40}\n")

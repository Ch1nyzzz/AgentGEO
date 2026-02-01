import logging
from geo_agent.config import get_search_client_from_config, load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

class SearchManager:
    """
    Manages search engine selection and execution.
    """
    def __init__(self, config_path='geo_agent/config.yaml'):
        from geo_agent.retrieval import ContentLoader
        self.config = load_config(config_path)
        self.search_provider = get_search_client_from_config(config_path)
        self.content_loader = ContentLoader(config_path)
        self._content_cache = {}


    def search(self, query: str):
        """
        Executes search using the configured provider.
        Returns a list of search results (metadata only).
        """
        try:
            results = self.search_provider.search(query)
            # Filter empty URLs
            valid_results = [r for r in results if r.url]
            self.results = valid_results
            return valid_results
        except Exception as e:
            logging.error(f"Search error: {e}")
            return []

    def search_and_retrieve(self, query: str):
        """
        Combines search and retrieval into one step.
        Returns a list of contents.
        """
        results = self.search(query)
        
        contents = []
        for doc in results:
            cache_key = doc.uuid if doc.uuid else doc.url
            if cache_key in self._content_cache:
                contents.append(self._content_cache[cache_key])
            else:
                content = self.content_loader.process(doc)
                if content:
                    self._content_cache[cache_key] = content
                    contents.append(content)
                else:
                    contents.append(doc.snippet or "")
        logging.info(f"Fetched contents for {len(results)} competitor documents.")
        return contents
    


import hashlib
import os
from typing import List

from geo_agent.core.models import SearchResult

class TavilyClient:
    def __init__(self, max_results: int = 5):
        self.max_results = max_results

    def search(self, query: str) -> List[SearchResult]:
        try:
            from tavily import TavilyClient as TavilySDKClient

            api_key = os.getenv("TAVILY_API_KEY", "")
            if not api_key:
                raise ValueError("Missing TAVILY_API_KEY environment variable.")

            client = TavilySDKClient(api_key=api_key)
            resp = client.search(
                query=query,
                search_depth="advanced",
                max_results=self.max_results,
                include_raw_content=True,
            )
            results = resp.get("results", []) if isinstance(resp, dict) else []
            doc_list = []
            for i, result in enumerate(results):
                # Tavily returns: {'url': ..., 'content': ..., 'title': ..., 'raw_content': ...}

                hash_object = hashlib.md5(result.get('url', '').encode())
                uuid = hash_object.hexdigest()

                doc = SearchResult(
                    idx=i,
                    title=result.get('title') or '',
                    url=result.get('url') or '',
                    snippet=result.get('content') or '',
                    uuid=uuid,
                    raw_content=result.get('raw_content') or '',
                )
                doc_list.append(doc)
            return doc_list
        except Exception as e:
            print(f"Tavily search error: {e}")
            return []

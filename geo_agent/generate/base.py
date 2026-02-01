from abc import ABC, abstractmethod
from typing import List
from geo_agent.core import WebPage, SearchResult, CitationCheckResult

class BaseGenerator(ABC):
    @abstractmethod
    def generate_and_check(self, query: str, target_doc: WebPage, retrieved_docs: List[SearchResult]) -> CitationCheckResult:
        """
        Generates an answer based on the query and retrieved documents, 
        and checks if the target document is cited.
        """
        pass

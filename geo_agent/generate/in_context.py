import re
import logging
import random
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from geo_agent.core import WebPage, CitationCheckResult
from geo_agent.config import get_llm_from_config
from geo_agent.retrieval import ContentLoader
from .base import BaseGenerator
from geo_agent.config import load_config

logger = logging.getLogger(__name__)

class InContextGenerator(BaseGenerator):
    def __init__(self, config_path='geo_agent/config.yaml'):
        self.llm = get_llm_from_config(config_path)
        gen_config = load_config(config_path).get('generator', {})
        self.max_snippet_length = gen_config.get('max_snippet_length', 4000)
        self.content_loader = ContentLoader(config_path)

    def generate(self, query: str, documents: List[str]) -> str:
        """
        Generates an answer based on the query and provided documents.
        """
        sources_text = ""
        for i, doc in enumerate(documents):
            doc_content = doc[:self.max_snippet_length]
            sources_text += f"[Document {i+1}:] \nContent: {doc_content}\n\n"

        messages = [
            ("system", """Write an accurate and concise answer for the given user question, using _only_ the provided summarized web search results. The answer should be correct, high-quality, and written by an expert using an unbiased and journalistic tone. The user’s language of choice such as English, Français, Español, Deutsch, or Japanese should be used. The answer should be informative, interesting, and engaging. The answer’s logic and reasoning should be rigorous and defensible. Every sentence in the answer should be _immediately followed_ by an in-line citation to the search result(s). The cited search result(s) should fully support _all_ the information in the sentence. Search results need to be cited using [index]. When citing several search results, use [1][2][3] format rather than [1, 2, 3]. You can use multiple search results to respond comprehensively while avoiding irrelevant search results."""),
            ("human", """Question: {query} \n\n Search Results: {sources}""")
        ]
        
        prompt = ChatPromptTemplate.from_messages(messages)

        try:
            response = self.llm.invoke(prompt.format_messages(query=query, sources=sources_text))
            answer = response.content
            logger.info(f"Generated answer length: {len(answer)}")
            return answer
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return "Error generating answer."

    async def agenerate(self, query: str, documents: List[str]) -> str:
        """
        Generates an answer based on the query and provided documents asynchronously.
        """
        sources_text = ""
        for i, doc in enumerate(documents):
            doc_content = doc[:self.max_snippet_length]
            sources_text += f"[Document {i+1}:] \nContent: {doc_content}\n\n"

        messages = [
            ("system", """Write an accurate and concise answer for the given user question, using _only_ the provided summarized web search results. The answer should be correct, high-quality, and written by an expert using an unbiased and journalistic tone. The user’s language of choice such as English, Français, Español, Deutsch, or Japanese should be used. The answer should be informative, interesting, and engaging. The answer’s logic and reasoning should be rigorous and defensible. Every sentence in the answer should be _immediately followed_ by an in-line citation to the search result(s). The cited search result(s) should fully support _all_ the information in the sentence. Search results need to be cited using [index]. When citing several search results, use [1][2][3] format rather than [1, 2, 3]. You can use multiple search results to respond comprehensively while avoiding irrelevant search results."""),
            ("human", """Question: {query} \n\n Search Results: {sources}""")
        ]
        
        prompt = ChatPromptTemplate.from_messages(messages)

        try:
            response = await self.llm.ainvoke(prompt.format_messages(query=query, sources=sources_text))
            answer = response.content
            logger.info(f"Generated answer length: {len(answer)}")
            return answer
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return "Error generating answer."

    def generate_and_check(self, query: str, target_doc: WebPage, competitors: List[str]) -> CitationCheckResult:
        """
        Uses LLM to simulate a search engine's generation process and checks if Target Doc is cited.
        """
        all_docs = competitors + [target_doc.cleaned_content]
        answer = self.generate(query, all_docs)
        target_idx = len(all_docs)  # Target doc is the last one
        return self._check_citation(answer, target_idx)

    
    def get_cited_indices(self, answer: str) -> List[int]:
        """
        Returns all document indices cited in the generated answer.
        """
        return list(set([int(m) for m in re.findall(r"\[(\d+)\]", answer)]))
    
    def _check_citation(self, answer: str, target_idx: int) -> CitationCheckResult:
        """
        Checks if the target document is cited in the generated answer using [index] format.
        """
        # Check if target doc index or url is present in the answer
        is_cited = f"[{target_idx}]" in answer

        # Extract cited sources
        cited_indices = self.get_cited_indices(answer)
        
        return CitationCheckResult(is_cited=is_cited, generated_answer=answer, citations_found_idx=cited_indices)
    
    def get_one_cited_content(self, answer: str, competitors: List[str]) -> str:
        """
        Returns the content of one cited document from the generated answer.
        Prefers competitors over target doc if both are cited.
        """
        cited_indices = self.get_cited_indices(answer)
        valid_indices = [idx for idx in cited_indices if 1 <= idx <= len(competitors)]
        
        if valid_indices:
            # Randomly select one to avoid always picking the first one
            selected_idx = random.choice(valid_indices)
            return competitors[selected_idx - 1]
            
        return ""

if __name__ == "__main__":
    # Example usage
    generator = InContextGenerator()

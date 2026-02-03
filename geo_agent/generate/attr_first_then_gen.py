import re
import logging
import random
import asyncio
from typing import List
from geo_agent.core import WebPage, SearchResult, CitationCheckResult
from geo_agent.config import get_llm_from_config
from geo_agent.retrieval import ContentLoader
from .base import BaseGenerator
from geo_agent.config import load_config

logger = logging.getLogger(__name__)

class AttrFirstThenGenerate(BaseGenerator):
    """
    Generator that uses attribute-first-then-generate approach for answer generation.
    """
    def __init__(self, config_path='geo_agent/config.yaml'):
        gen_config = load_config(config_path).get('generator', {})
        
        self.llm = get_llm_from_config(config_path)
        self.max_snippet_length = gen_config.get('max_snippet_length', 4000)
        self.content_loader = ContentLoader(config_path)
        logger.info("Initialized AttrFirstThenGenerate")
        
        self.attr_evaluator = None
        try:
            # Import fast_return_res
            from attr_evaluator.fast_return_res import fast_return_res
            
            self.attribute_return_res = fast_return_res
            logger.info("Successfully initialized fast_return_res")
        except Exception as e:
            logger.error(f"Failed to initialize fast_return_res: {e}")
            raise

    def generate(self, query: str, documents: List[str]) -> str:
        """
        Generates an answer based on the query and provided documents using attr_evaluator.
        """
        try:
            # Prepare documents for attr_evaluator (limit length)
            rewritten_docs = [doc[:self.max_snippet_length] for doc in documents]
            
            # Call fast_return_res function
            result = self.attribute_return_res(
                id="test_id",
                query=query,
                rewritten_docs=rewritten_docs
            )
            
            if "answer_prompt" in result:
                answer = result["answer_prompt"]
                logger.info(f"Generated answer length using attr_evaluator: {len(answer)}")
                return answer
            else:
                logger.error("attr_evaluator returned no answer_prompt")
                return "Error: attr_evaluator returned no answer_prompt"
        except Exception as e:
            logger.error(f"attr_evaluator generation failed: {e}")
            return "Error generating answer with attr_evaluator."

    async def agenerate(self, query: str, documents: List[str]) -> str:
        """
        Generates an answer based on the query and provided documents asynchronously.
        Wraps the synchronous attribute_return_res in a thread.
        """
        return await asyncio.to_thread(self.generate, query, documents)

    def generate_and_check(self, query: str, target_doc: WebPage, competitors: List[str]) -> CitationCheckResult:
        """
        Uses attr_evaluator to generate answers and checks if Target Doc is cited.
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

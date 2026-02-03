"""
Pluggable Citation Checker Architecture

Provides unified interface for multiple citation checking methods:
- LLM: Generate answer with LLM and check citations (default method)
- AttrEvaluator: Use Attribute Evaluator precision pipeline
- Both: Use both methods and determine final result based on strategy
"""
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import asyncio
import logging
import re
import tempfile
from pathlib import Path

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)


@dataclass
class GEOScoreInfo:
    """GEO Score detailed information"""
    word: float = 0.0           # Word count weighted score
    position: float = 0.0       # Position weighted score
    wordpos: float = 0.0        # Word + position combined score
    overall: float = 0.0        # Final GEO Score = 0.33 * (wordpos + word + pos)
    target_idx: int = 0
    num_sources: int = 0
    has_valid_citations: bool = False  # Whether answer contains valid [n] format citations


class CitationMethod(str, Enum):
    """Citation checking method enum"""
    LLM = "llm"
    ATTR_EVALUATOR = "attr_evaluator"
    BOTH = "both"


@dataclass
class CitationResult:
    """Unified citation check result"""
    is_cited: bool
    generated_answer: str
    citations_found_idx: List[int]
    method: CitationMethod = CitationMethod.LLM

    # LLM method specific fields
    cited_by_index: bool = False
    cited_by_url: bool = False

    # Attribute Evaluator method specific fields
    attr_citations: Optional[List[int]] = None  # [1,0,1,...] format
    highlight_set: Optional[List[Dict]] = None

    # Composite result (both mode)
    llm_result: Optional["CitationResult"] = None
    attr_result: Optional["CitationResult"] = None

    # GEO Score (V2.3 addition)
    geo_score: Optional[GEOScoreInfo] = None


def compute_geo_score(
    generated_answer: str,
    target_idx: int,
    num_sources: int
) -> GEOScoreInfo:
    """
    Compute GEO Score

    Args:
        generated_answer: LLM generated answer containing [n] format citations
        target_idx: Target document index (1-based)
        num_sources: Total number of documents

    Returns:
        GEOScoreInfo: GEO Score detailed information
    """
    from autogeo.evaluation.metrics.geo_score import (
        extract_citations_new,
        impression_word_count_simple,
        impression_pos_count_simple,
        impression_wordpos_count_simple,
    )

    # Check if there are valid [n] format citations
    citation_pattern = r'\[[^\w\s]*\d+[^\w\s]*\]'
    has_valid_citations = bool(re.search(citation_pattern, generated_answer))

    if not has_valid_citations or not generated_answer.strip():
        return GEOScoreInfo(
            word=0.0,
            position=0.0,
            wordpos=0.0,
            overall=0.0,
            target_idx=target_idx,
            num_sources=num_sources,
            has_valid_citations=False,
        )

    try:
        # Extract citation structure
        sentences = extract_citations_new(generated_answer)

        # Calculate three types of impression scores
        word_scores = impression_word_count_simple(sentences, n=num_sources, normalize=True)
        pos_scores = impression_pos_count_simple(sentences, n=num_sources, normalize=True)
        wordpos_scores = impression_wordpos_count_simple(sentences, n=num_sources, normalize=True)

        # Get target document score (target_idx is 1-based, convert to 0-based)
        idx = target_idx - 1
        if 0 <= idx < len(word_scores):
            word_score = word_scores[idx]
            pos_score = pos_scores[idx]
            wordpos_score = wordpos_scores[idx]
        else:
            word_score = 0.0
            pos_score = 0.0
            wordpos_score = 0.0

        # Calculate combined GEO Score
        overall = (wordpos_score + word_score + pos_score) / 3.0

        return GEOScoreInfo(
            word=word_score,
            position=pos_score,
            wordpos=wordpos_score,
            overall=overall,
            target_idx=target_idx,
            num_sources=num_sources,
            has_valid_citations=True,
        )
    except Exception as e:
        logger.warning(f"Failed to compute GEO Score: {e}")
        return GEOScoreInfo(
            word=0.0,
            position=0.0,
            wordpos=0.0,
            overall=0.0,
            target_idx=target_idx,
            num_sources=num_sources,
            has_valid_citations=has_valid_citations,
        )


def compute_geo_score_from_citations(
    citations: List[int],
    target_idx: int,
    num_sources: int
) -> GEOScoreInfo:
    """
    Compute simplified GEO Score from binary citations list

    When there are no standard [n] format citations, use citations list returned by AttrEvaluator
    This is a simplified version that can only determine if cited, cannot compute position and word weights

    Args:
        citations: Binary citation list [0,1,0,...] indicating whether document i is cited
        target_idx: Target document index (1-based)
        num_sources: Total number of documents

    Returns:
        GEOScoreInfo: Simplified GEO Score
    """
    # Calculate total number of cited documents
    cited_count = sum(citations) if citations else 0

    # Check if target document is cited
    idx = target_idx - 1
    is_target_cited = (0 <= idx < len(citations) and citations[idx] == 1)

    if not is_target_cited or cited_count == 0:
        return GEOScoreInfo(
            word=0.0,
            position=0.0,
            wordpos=0.0,
            overall=0.0,
            target_idx=target_idx,
            num_sources=num_sources,
            has_valid_citations=False,
        )

    # Simplified: assume all cited documents share scores equally
    score = 1.0 / cited_count

    return GEOScoreInfo(
        word=score,
        position=score,
        wordpos=score,
        overall=score,
        target_idx=target_idx,
        num_sources=num_sources,
        has_valid_citations=True,  # No [n] format, but has valid citations
    )


class BaseCitationChecker(ABC):
    """Citation checker abstract base class"""

    @abstractmethod
    async def check(
        self,
        query: str,
        target_content: str,
        target_url: str,
        target_idx: int,
        retrieved_docs: List[Any],
        competitor_contents: List[str],
    ) -> CitationResult:
        """
        Check if target content is cited

        Args:
            query: Query question
            target_content: Target document content
            target_url: Target document URL
            target_idx: Target document index in result list (1-based)
            retrieved_docs: Retrieved competitor documents list
            competitor_contents: Competitor document contents list

        Returns:
            CitationResult: Citation check result
        """
        pass


class LLMCitationChecker(BaseCitationChecker):
    """LLM citation checker (current implementation)"""

    def __init__(self, llm: "BaseChatModel", max_snippet_length: int = 2000):
        self.llm = llm
        self.max_snippet_length = max_snippet_length

    async def _ainvoke(self, prompt_input):
        if hasattr(self.llm, "ainvoke"):
            return await self.llm.ainvoke(prompt_input)
        return await asyncio.to_thread(self.llm.invoke, prompt_input)

    async def check(
        self,
        query: str,
        target_content: str,
        target_url: str,
        target_idx: int,
        retrieved_docs: List[Any],
        competitor_contents: List[str],
    ) -> CitationResult:
        from langchain_core.prompts import ChatPromptTemplate

        # Build sources text
        sources_text = ""
        for i, content in enumerate(competitor_contents):
            if not content:
                continue
            content = content[: self.max_snippet_length]
            sources_text += f"[Source {i + 1}] \nContent: {content}\n\n"

        # Add target document
        if not target_content or not target_content.strip():
            logger.warning("target_content is empty, returning not cited")
            return CitationResult(
                is_cited=False,
                generated_answer="",
                citations_found_idx=[],
                method=CitationMethod.LLM,
            )

        target_content_trimmed = target_content[: self.max_snippet_length]
        sources_text += f"[Source {target_idx}] \nContent: {target_content_trimmed}\n"

        messages = [
            (
                "system",
                "Write an accurate and concise answer for the given user question, "
                "using _only_ the provided summarized web search results. The answer "
                "should be correct, high-quality, and written by an expert using an "
                "unbiased and journalistic tone. The user's language of choice such as "
                "English, Francais, Espanol, Deutsch, or Japanese should be used. The "
                "answer should be informative, interesting, and engaging. The answer's "
                "logic and reasoning should be rigorous and defensible. Every sentence "
                "in the answer should be _immediately followed_ by an in-line citation "
                "to the search result(s). The cited search result(s) should fully support "
                "_all_ the information in the sentence. Search results need to be cited "
                "using [index]. When citing several search results, use [1][2][3] format "
                "rather than [1, 2, 3]. You can use multiple search results to respond "
                "comprehensively while avoiding irrelevant search results.",
            ),
            (
                "human",
                "Question: {query}\n\nSearch Results:\n{sources}",
            ),
        ]

        prompt = ChatPromptTemplate.from_messages(messages)

        try:
            response = await self._ainvoke(prompt.format(query=query, sources=sources_text))
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            raise RuntimeError("LLM generation failed.") from exc

        answer = response.content

        # Check citations
        cited_by_index = f"[{target_idx}]" in answer
        cited_by_url = bool(target_url) and target_url in answer
        is_cited = cited_by_index or cited_by_url
        cited_indices = list({int(m) for m in re.findall(r"\[(\d+)\]", answer)})

        # Calculate GEO Score (LLM answer has standard [n] format)
        num_sources = len(competitor_contents) + 1  # 竞争文档 + 目标文档
        geo_score_info = compute_geo_score(answer, target_idx, num_sources)

        return CitationResult(
            is_cited=is_cited,
            generated_answer=answer,
            citations_found_idx=cited_indices,
            method=CitationMethod.LLM,
            cited_by_index=cited_by_index,
            cited_by_url=cited_by_url,
            geo_score=geo_score_info,
        )


class AttrEvaluatorCitationChecker(BaseCitationChecker):
    """Attribute Evaluator citation checker"""

    def __init__(self, config_file: Optional[str] = None, use_fast_mode: bool = True):
        self.config_file = config_file
        self.use_fast_mode = use_fast_mode

    async def check(
        self,
        query: str,
        target_content: str,
        target_url: str,
        target_idx: int,
        retrieved_docs: List[Any],
        competitor_contents: List[str],
    ) -> CitationResult:
        # Build document list: competitor docs + target doc (insert at target_idx-1 position)
        all_docs = list(competitor_contents)
        insert_pos = target_idx - 1
        if insert_pos < 0:
            insert_pos = 0
        if insert_pos > len(all_docs):
            insert_pos = len(all_docs)
        all_docs.insert(insert_pos, target_content)
        
        # Output search result info
        logger.info(f"\n=== AttrEvaluatorCitationChecker starting citation check ===")
        
        try:
            if self.use_fast_mode:
                logger.info(f"Using fast_return_res for citation check")
                from attr_evaluator.fast_return_res import fast_return_res
                # Call fast_return_res (run in thread to avoid blocking)
                result = await asyncio.to_thread(
                    fast_return_res, "test_id", query, all_docs, self.config_file
                )
            else:
                logger.info(f"Using original return_res for citation check")
                from attr_evaluator.run_dataset import return_res, pre_init
                # Initialize args
                arg_list = []
                if self.config_file:
                    arg_list.extend(["--config-file", self.config_file])
                args = pre_init(arg_list)
                # Call return_res (run in thread to avoid blocking)
                result = await asyncio.to_thread(
                    return_res, "test_id", query, all_docs, args
                )
        except Exception as e:
            logger.error("AttrEvaluator check failed: %s", e)
            import traceback
            logger.error("Traceback: %s", traceback.format_exc())
            return CitationResult(
                is_cited=False,
                generated_answer="",
                citations_found_idx=[],
                method=CitationMethod.ATTR_EVALUATOR,
                attr_citations=[],
                highlight_set=[],
            )

        citations = result.get("citations", [])
        logger.info(f"Citation result: {citations}")
        
        # Check if target document is cited (target_idx is 1-based)
        is_cited = False
        if target_idx <= len(citations):
            is_cited = citations[target_idx - 1] == 1
            logger.info(f"Target document cited: {'✅ Yes' if is_cited else '❌ No'}")

        # Collect all cited document indices (convert to 1-based)
        citations_found_idx = [i + 1 for i, c in enumerate(citations) if c == 1]
        logger.info(f"All cited document indices (1-based): {citations_found_idx}")
        
        generated_answer = result.get("answer_prompt", "")
        # logger.info(f"Generated answer: {generated_answer[:100]}...")
        # logger.info(f"Highlight set size: {len(result.get('highlight_set', []))}")
        logger.info(f"=== AttrEvaluatorCitationChecker citation check complete ===")

        # Calculate GEO Score
        num_sources = len(all_docs)
        # Check if generated answer has [n] format citations
        citation_pattern = r'\[[^\w\s]*\d+[^\w\s]*\]'
        has_citation_format = bool(re.search(citation_pattern, generated_answer))

        if has_citation_format:
            # Use standard calculation method
            geo_score_info = compute_geo_score(generated_answer, target_idx, num_sources)
        else:
            # Use simplified version (based on citations list)
            geo_score_info = compute_geo_score_from_citations(citations, target_idx, num_sources)

        return CitationResult(
            is_cited=is_cited,
            generated_answer=generated_answer,
            citations_found_idx=citations_found_idx,
            method=CitationMethod.ATTR_EVALUATOR,
            attr_citations=citations,
            highlight_set=result.get("highlight_set", []),
            geo_score=geo_score_info,
        )


class CompositeCitationChecker(BaseCitationChecker):
    """
    Composite citation checker (uses both methods simultaneously)

    Strategy description:
    - any: Considered cited if either method detects citation
    - all: Considered cited only if both methods detect citation
    - llm_primary: LLM result is primary, attr result is supplementary
    - attr_primary: AttrEvaluator result is primary, LLM result is supplementary
    """

    def __init__(
        self,
        llm_checker: LLMCitationChecker,
        attr_checker: AttrEvaluatorCitationChecker,
        strategy: str = "any",
    ):
        self.llm_checker = llm_checker
        self.attr_checker = attr_checker
        self.strategy = strategy

    async def check(
        self,
        query: str,
        target_content: str,
        target_url: str,
        target_idx: int,
        retrieved_docs: List[Any],
        competitor_contents: List[str],
    ) -> CitationResult:
        # Execute both checks in parallel
        llm_task = self.llm_checker.check(
            query, target_content, target_url, target_idx,
            retrieved_docs, competitor_contents
        )
        attr_task = self.attr_checker.check(
            query, target_content, target_url, target_idx,
            retrieved_docs, competitor_contents
        )

        llm_result, attr_result = await asyncio.gather(
            llm_task, attr_task, return_exceptions=True
        )

        # Handle exceptions
        if isinstance(llm_result, Exception):
            logger.error("LLM checker failed: %s", llm_result)
            llm_result = CitationResult(
                is_cited=False, generated_answer="", citations_found_idx=[],
                method=CitationMethod.LLM
            )
        if isinstance(attr_result, Exception):
            logger.error("AttrEvaluator checker failed: %s", attr_result)
            attr_result = CitationResult(
                is_cited=False, generated_answer="", citations_found_idx=[],
                method=CitationMethod.ATTR_EVALUATOR
            )

        # Determine final result based on strategy
        if self.strategy == "any":
            is_cited = llm_result.is_cited or attr_result.is_cited
        elif self.strategy == "all":
            is_cited = llm_result.is_cited and attr_result.is_cited
        elif self.strategy == "llm_primary":
            is_cited = llm_result.is_cited
        elif self.strategy == "attr_primary":
            is_cited = attr_result.is_cited
        else:
            logger.warning("Unknown strategy '%s', defaulting to 'any'", self.strategy)
            is_cited = llm_result.is_cited or attr_result.is_cited

        # Merge citation indices
        all_citations = set(llm_result.citations_found_idx) | set(attr_result.citations_found_idx)

        # Prefer LLM generated answer
        generated_answer = llm_result.generated_answer or attr_result.generated_answer

        # Merge GEO Score based on strategy
        llm_geo = llm_result.geo_score
        attr_geo = attr_result.geo_score

        if self.strategy == "llm_primary":
            merged_geo_score = llm_geo
        elif self.strategy == "attr_primary":
            merged_geo_score = attr_geo
        elif self.strategy in ("any", "all"):
            # Prefer the one with valid citations, take average if both have
            if llm_geo and attr_geo and llm_geo.has_valid_citations and attr_geo.has_valid_citations:
                merged_geo_score = GEOScoreInfo(
                    word=(llm_geo.word + attr_geo.word) / 2,
                    position=(llm_geo.position + attr_geo.position) / 2,
                    wordpos=(llm_geo.wordpos + attr_geo.wordpos) / 2,
                    overall=(llm_geo.overall + attr_geo.overall) / 2,
                    target_idx=llm_geo.target_idx,
                    num_sources=llm_geo.num_sources,
                    has_valid_citations=True,
                )
            elif llm_geo and llm_geo.has_valid_citations:
                merged_geo_score = llm_geo
            elif attr_geo and attr_geo.has_valid_citations:
                merged_geo_score = attr_geo
            else:
                merged_geo_score = llm_geo or attr_geo
        else:
            merged_geo_score = llm_geo or attr_geo

        return CitationResult(
            is_cited=is_cited,
            generated_answer=generated_answer,
            citations_found_idx=sorted(all_citations),
            method=CitationMethod.BOTH,
            cited_by_index=llm_result.cited_by_index,
            cited_by_url=llm_result.cited_by_url,
            attr_citations=attr_result.attr_citations,
            highlight_set=attr_result.highlight_set,
            llm_result=llm_result,
            attr_result=attr_result,
            geo_score=merged_geo_score,
        )


def create_citation_checker(
    method: CitationMethod,
    llm: Optional["BaseChatModel"] = None,
    max_snippet_length: int = 2000,
    attr_evaluator_config: Optional[str] = None,
    composite_strategy: str = "any",
    use_fast_mode: bool = True,
) -> BaseCitationChecker:
    """
    Factory function: Create citation checker

    Args:
        method: Citation checking method
        llm: LLM instance (required for LLM and BOTH modes)
        max_snippet_length: Maximum content snippet length
        attr_evaluator_config: AttrEvaluator config file path
        composite_strategy: Composite mode strategy
        use_fast_mode: Whether to use fast mode, default True

    Returns:
        BaseCitationChecker: Citation checker instance
    """
    if method == CitationMethod.LLM:
        if llm is None:
            raise ValueError("LLM instance is required for LLM citation method")
        return LLMCitationChecker(llm, max_snippet_length)

    elif method == CitationMethod.ATTR_EVALUATOR:
        return AttrEvaluatorCitationChecker(attr_evaluator_config, use_fast_mode)

    elif method == CitationMethod.BOTH:
        if llm is None:
            raise ValueError("LLM instance is required for BOTH citation method")
        llm_checker = LLMCitationChecker(llm, max_snippet_length)
        attr_checker = AttrEvaluatorCitationChecker(attr_evaluator_config, use_fast_mode)
        return CompositeCitationChecker(llm_checker, attr_checker, composite_strategy)

    else:
        raise ValueError(f"Unknown citation method: {method}")

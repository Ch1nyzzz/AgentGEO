"""
AgentGEO optimizer wrapper
"""
import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Add project path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from geo_agent.batch_suggestion_orchestrator import AgentGEOV2, AgentGEOConfigV2
from geo_agent.core.models import WebPage
from geo_agent.utils.structural_parser import StructuralHtmlParser

logger = logging.getLogger(__name__)


@dataclass
class AgentGEOResult:
    """AgentGEO optimization result"""
    optimized_text: str
    optimized_html: str
    optimization_results: List
    original_text: str
    diagnosis_stats: dict


class AgentGEOOptimizer:
    """
    AgentGEO optimizer wrapper

    Uses StructuralHtmlParser uniformly to parse HTML
    """

    def __init__(
        self,
        config_path: str = "geo_agent/config.yaml",
        batch_size: int = 10,
        max_concurrency: int = 4,
        citation_method: str = "llm",
        enable_memory: bool = True,
        enable_history: bool = True,
        **kwargs
    ):
        self.config_path = config_path
        self.batch_config = AgentGEOConfigV2(
            batch_size=batch_size,
            max_concurrency=max_concurrency,
            citation_method=citation_method,
            enable_memory=enable_memory,
            enable_history=enable_history,
            **kwargs
        )
        self._parser = StructuralHtmlParser()
        self._agent: Optional[AgentGEOV2] = None

    def _get_agent(self) -> AgentGEOV2:
        """Get or create AgentGEOV2 instance"""
        if self._agent is None:
            self._agent = AgentGEOV2(
                config_path=self.config_path,
                batch_config=self.batch_config,
            )
            logger.info("AgentGEOV2 initialized")
        return self._agent

    async def optimize_async(
        self,
        raw_html: str,
        train_queries: List[str],
        url: str = ""
    ) -> AgentGEOResult:
        """Async document optimization"""
        agent = self._get_agent()

        # Extract original text
        structure = self._parser.parse(raw_html)
        original_text = structure.get_clean_text()

        # Create WebPage
        webpage = WebPage(
            url=url,
            raw_html=raw_html,
            cleaned_content=original_text
        )

        logger.info(f"Optimizing with AgentGEO (queries: {len(train_queries)})")

        # Call optimization
        optimized_page, results = await agent.optimize_page_batch_async(
            webpage=webpage,
            queries=train_queries
        )

        # Aggregate diagnosis statistics
        diagnosis_stats = {}
        for result in results:
            for cause, count in result.diagnosis_stats.items():
                diagnosis_stats[cause] = diagnosis_stats.get(cause, 0) + count

        return AgentGEOResult(
            optimized_text=optimized_page.cleaned_content,
            optimized_html=optimized_page.raw_html,
            optimization_results=results,
            original_text=original_text,
            diagnosis_stats=diagnosis_stats
        )

    def optimize(self, raw_html: str, train_queries: List[str], url: str = "") -> AgentGEOResult:
        """Sync optimization interface (internally calls async method)"""
        return asyncio.run(self.optimize_async(raw_html, train_queries, url))

    async def evaluate_page_async(
        self,
        raw_html: str,
        test_queries: List[str],
        url: str = ""
    ) -> dict:
        """
        Evaluate page citation rate using test queries (held-out evaluation)

        Args:
            raw_html: HTML content
            test_queries: Test queries for evaluation
            url: Page URL

        Returns:
            Dict with 'ratio' (citation rate) and per-query results
        """
        agent = self._get_agent()

        # Parse HTML and create WebPage
        structure = self._parser.parse(raw_html)
        text = structure.get_clean_text()
        webpage = WebPage(
            url=url,
            raw_html=raw_html,
            cleaned_content=text
        )

        logger.info(f"Evaluating with {len(test_queries)} test queries")
        result = await agent.evaluate_page_async(webpage, test_queries)
        return result

"""
AgentGEO V2 - Main Entry Point
Intelligent GEO optimization system based on suggestion orchestration

Key Features:
1. Two-phase failure analysis from geo_agent (diagnose + select_tool_strategy)
2. Support for 14 failure categories (FAILURE_TAXONOMY)
3. Async parallel processing of multiple queries
4. HTML DOM chunking support (StructuralHtmlParser)
5. Intelligent suggestion merging (diagnosis-aware)
6. Complete history tracking and policy injection support
"""
import asyncio
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Setup paths
REPO_ROOT = Path(__file__).resolve().parents[2]
GEO_AGENT_ROOT = REPO_ROOT / "geo_agent"
if str(GEO_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(GEO_AGENT_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

from geo_agent.config import get_llm_from_config, get_llm_for_task, load_config, LLMTask
from geo_agent.core.models import WebPage, SearchResult, CitationCheckResult
from geo_agent.search_engine.manager import SearchManager
from geo_agent.search_engine.chatnoir import ChatNoirClient
from geo_agent.utils.html_parser import HtmlParser

from .suggestion_processor import SuggestionProcessorV2
from .memory_manager import HistoryManagerV2
from .models import AgentGEOConfigV2, OptimizationResultV2
from .citation_checker import (
    BaseCitationChecker,
    CitationMethod,
    CitationResult,
    LLMCitationChecker,
    create_citation_checker,
)

logger = logging.getLogger(__name__)


class AsyncInContextGeneratorV2:
    """
    V2 Async Generator

    Supports pluggable citation checker architecture, allowing different check methods via citation_checker parameter.
    """

    def __init__(
        self,
        config_path: str = "geo_agent/config.yaml",
        max_snippet_length: int = 2000,
        citation_checker: Optional[BaseCitationChecker] = None,
    ):
        # Use LLM configured for GENERATION task
        self.llm = get_llm_for_task(config_path, LLMTask.GENERATION)
        self.max_snippet_length = max_snippet_length

        # If not provided, create default LLM checker (backward compatible)
        if citation_checker is None:
            self._citation_checker = LLMCitationChecker(self.llm, max_snippet_length)
        else:
            self._citation_checker = citation_checker

    async def generate_and_check(
        self,
        query: str,
        target_doc: WebPage,
        retrieved_docs: List[SearchResult],
        competitor_contents: Optional[List[str]] = None,
    ) -> CitationCheckResult:
        """
        Generate answer and check citations (delegated to citation_checker)

        Args:
            query: Query question
            target_doc: Target document
            retrieved_docs: Retrieved competitor documents
            competitor_contents: Competitor document content list

        Returns:
            CitationCheckResult: Citation check result (backward compatible)
        """
        num_competitors = len(retrieved_docs)
        target_idx = num_competitors + 1

        if competitor_contents is None or len(competitor_contents) < len(retrieved_docs):
            raise ValueError("competitor_contents is required")

        # Delegate to citation_checker
        result = await self._citation_checker.check(
            query=query,
            target_content=target_doc.cleaned_content or "",
            target_url=target_doc.url or "",
            target_idx=target_idx,
            retrieved_docs=retrieved_docs,
            competitor_contents=competitor_contents,
        )

        # Return CitationCheckResult for backward compatibility
        return CitationCheckResult(
            is_cited=result.is_cited,
            generated_answer=result.generated_answer,
            citations_found_idx=result.citations_found_idx,
            geo_score=result.geo_score,
        )

    @property
    def citation_checker(self) -> BaseCitationChecker:
        """Get the current citation checker"""
        return self._citation_checker


class AgentGEOV2:
    """
    AgentGEO V2 - Intelligent GEO Optimization System based on Suggestion Orchestration

    Async parallel batch processing version fully based on geo_agent architecture

    Key Features:
    - Two-phase failure analysis (diagnose + select_tool_strategy)
    - 14 failure categories
    - Async parallel processing
    - HTML DOM chunking
    - Intelligent suggestion merging
    - Policy injection support
    """

    def __init__(
        self,
        config_path: str = "geo_agent/config.yaml",
        batch_config: Optional[AgentGEOConfigV2] = None,
        disk_cache_dir: Optional[str] = None,
        optimization_log_dir: Optional[str] = None,
        output_prefix: str = "",
        run_id: Optional[str] = None,
    ):
        config_file = Path(config_path)
        if not config_file.is_absolute():
            config_file = (REPO_ROOT / config_file).resolve()
        self.config_path = str(config_file)

        # Config (initialize first, needed for creating citation_checker later)
        self.batch_config = batch_config or AgentGEOConfigV2()

        # LLM and search engine
        # Main LLM used for diagnosis analysis
        self.llm = get_llm_for_task(self.config_path, LLMTask.DIAGNOSIS)
        self.search_engine = SearchManager(self.config_path)

        # Create citation checker
        generation_llm = get_llm_for_task(self.config_path, LLMTask.GENERATION)
        self._citation_checker = create_citation_checker(
            method=CitationMethod(self.batch_config.citation_method),
            llm=generation_llm,
            max_snippet_length=2000,
            attr_evaluator_config=self.batch_config.attr_evaluator_config,
            composite_strategy=self.batch_config.citation_composite_strategy,
            use_fast_mode=self.batch_config.use_fast_mode,
        )

        # Generator uses independent GENERATION config and citation_checker
        self.generator = AsyncInContextGeneratorV2(
            self.config_path,
            citation_checker=self._citation_checker,
        )

        # Search engine type
        self.config = load_config(self.config_path)
        search_config = self.config.get("search", {})
        self._search_provider = search_config.get("provider", "chatnoir")

        if self._search_provider == "chatnoir":
            self._chatnoir_client = ChatNoirClient()
        else:
            self._chatnoir_client = None

        self._html_parser = HtmlParser(self.config_path)

        # Cache
        self._html_content_cache: Dict[str, str] = {}
        self._search_cache: Dict[str, List[SearchResult]] = {}
        self._cache_lock = asyncio.Lock()

        # Disk cache
        self._disk_cache_dir: Optional[Path] = None
        if disk_cache_dir:
            cache_path = Path(disk_cache_dir)
            if not cache_path.is_absolute():
                cache_path = (REPO_ROOT / cache_path).resolve()
            cache_path.mkdir(parents=True, exist_ok=True)
            self._disk_cache_dir = cache_path

        # Log directory
        self._optimization_log_dir: Optional[Path] = None
        if optimization_log_dir:
            log_path = Path(optimization_log_dir)
            if not log_path.is_absolute():
                log_path = (REPO_ROOT / log_path).resolve()
            self._optimization_log_dir = log_path

        self._output_prefix = output_prefix
        self._run_id = run_id or os.getenv("DASHBOARD_RUN_ID")

        # History manager
        history_path = None
        if self.batch_config.history_persist_path:
            history_path = Path(self.batch_config.history_persist_path)
        self.history_manager = HistoryManagerV2(persist_path=history_path)

    def _cache_key(self, query: str) -> str:
        return hashlib.sha256(query.encode("utf-8")).hexdigest()

    def _disk_cache_path(self, query: str) -> Optional[Path]:
        if not self._disk_cache_dir:
            return None
        return self._disk_cache_dir / f"{self._cache_key(query)}.json"

    async def _get_search_results(self, query: str) -> List[SearchResult]:
        """Get search results (with caching)"""
        async with self._cache_lock:
            cached = self._search_cache.get(query)
        if cached is not None:
            return cached

        cache_path = self._disk_cache_path(query)
        if cache_path and cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                results = [SearchResult(**item) for item in payload]
                async with self._cache_lock:
                    self._search_cache[query] = results
                return results
            except Exception as exc:
                logger.warning("Failed to load disk cache: %s", exc)

        results = await asyncio.to_thread(self.search_engine.search, query)
        async with self._cache_lock:
            self._search_cache[query] = results

        if cache_path:
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump([r.model_dump() if hasattr(r, "model_dump") else r.dict() for r in results], f, ensure_ascii=False, indent=2)
            except Exception as exc:
                logger.warning("Failed to write disk cache: %s", exc)

        return results

    async def _get_competitor_full_content(self, doc: SearchResult) -> Optional[str]:
        """Get competitor full content (must fetch HTML, no fallback to snippet allowed)"""
        if not doc.uuid:
            raise RuntimeError(f"Competitor doc has no uuid, cannot fetch HTML: url={doc.url}")

        if doc.uuid in self._html_content_cache:
            return self._html_content_cache[doc.uuid]

        if self._search_provider == "tavily":
            if doc.raw_content:
                self._html_content_cache[doc.uuid] = doc.raw_content
                return doc.raw_content
            raise RuntimeError(f"Tavily doc has no raw_content, cannot fallback to snippet: uuid={doc.uuid}, url={doc.url}")
        else:
            # 1. First check if local dataset has HTML file for this UUID
            local_html_path = Path(self._disk_cache_dir) / f"{doc.uuid}.html" if self._disk_cache_dir else None
            if local_html_path and local_html_path.exists():
                try:
                    with open(local_html_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    logger.info(f"Read HTML from local disk cache: {doc.uuid}")
                    parsed_content = self._html_parser.parse(html_content)
                    self._html_content_cache[doc.uuid] = parsed_content
                    return parsed_content
                except Exception as e:
                    logger.warning(f"Failed to read local HTML for {doc.uuid}: {e}")
            
            # 2. Check original HTML database (cw22 mode)
            import os
            html_db_path = self.config.get("data", {}).get("html_db_path", "")
            if html_db_path:
                cw22_html_path = Path(html_db_path) / f"{doc.uuid}.html"
            else:
                cw22_html_path = None

            if cw22_html_path and cw22_html_path.exists():
                try:
                    with open(cw22_html_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    logger.info(f"Read HTML from cw22 dataset: {doc.uuid}")
                    parsed_content = self._html_parser.parse(html_content)
                    self._html_content_cache[doc.uuid] = parsed_content
                    return parsed_content
                except Exception as e:
                    logger.warning(f"Failed to read cw22 HTML for {doc.uuid}: {e}")
            
            # 3. If not available locally, fetch from ChatNoir API
            # Add retry mechanism to ensure sufficient time for execution
            import requests
            max_api_retries = 2
            for api_retry in range(max_api_retries):
                try:
                    html_content = await asyncio.to_thread(
                        self._chatnoir_client.get_html_content, doc.uuid
                    )
                    if html_content:
                        parsed_content = self._html_parser.parse(html_content)
                        self._html_content_cache[doc.uuid] = parsed_content
                        # Save locally for future use
                        if self._disk_cache_dir and local_html_path:
                            try:
                                local_html_path.parent.mkdir(parents=True, exist_ok=True)
                                with open(local_html_path, 'w', encoding='utf-8') as f:
                                    f.write(html_content)
                                logger.info(f"Saved HTML to local disk cache: {doc.uuid}")
                            except Exception as e:
                                logger.warning(f"Failed to save HTML to local cache for {doc.uuid}: {e}")
                        return parsed_content
                    else:
                        logger.warning(f"ChatNoir returned empty HTML for {doc.uuid}")
                        # If empty content returned, try with plain=True mode
                        if api_retry < max_api_retries - 1:
                            logger.info(f"Retrying with plain mode for {doc.uuid}...")
                            html_content = await asyncio.to_thread(
                                self._chatnoir_client.get_html_content, doc.uuid, plain=True
                            )
                            if html_content:
                                parsed_content = self._html_parser.parse(html_content)
                                self._html_content_cache[doc.uuid] = parsed_content
                                return parsed_content
                except Exception as e:
                    logger.warning(f"ChatNoir fetch failed for {doc.uuid} (attempt {api_retry+1}/{max_api_retries}): {e}")
                    # If connection error, wait and retry
                    if isinstance(e, (ConnectionResetError, requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.SSLError)) and api_retry < max_api_retries - 1:
                        logger.info(f"Waiting 3 seconds before retry for {doc.uuid}...")
                        await asyncio.sleep(3)  # 等待3秒后重试
                        continue
            
            # 4. All attempts failed, raise error
            raise RuntimeError(f"All attempts to fetch HTML failed for competitor: uuid={doc.uuid}, url={doc.url}")

    async def _get_all_competitor_contents(
        self, docs: List[SearchResult], target_count: int = 10
    ) -> Tuple[List[SearchResult], List[str]]:
        """Get all competitor contents"""
        filtered_docs = []
        filtered_contents = []

        for doc in docs:
            if len(filtered_contents) >= target_count:
                break

            content = await self._get_competitor_full_content(doc)
            if content:
                filtered_docs.append(doc)
                filtered_contents.append(content)

        return filtered_docs, filtered_contents

    async def optimize_page_batch_async(
        self,
        webpage: WebPage,
        queries: List[str],
    ) -> Tuple[WebPage, List[OptimizationResultV2]]:
        """
        Optimize page using batch processing

        Args:
            webpage: WebPage to optimize
            queries: List of queries

        Returns:
            Tuple[WebPage, List[OptimizationResultV2]]: (optimized webpage, batch results list)
        """
        print(f"Starting Batch GEO V2 for {webpage.url}")
        print("=" * 50)

        # Create batch processor
        processor = SuggestionProcessorV2(
            llm=self.llm,
            generator=self.generator,
            config=self.batch_config,
            history_manager=self.history_manager,
            search_func=self._get_search_results,
            competitor_content_func=self._get_all_competitor_contents,  # Returns tuple (filtered_docs, contents)
        )

        # Process all batches
        results = await processor.process_all_batches(
            content=webpage.cleaned_content,
            all_queries=queries,
            raw_html=webpage.raw_html if webpage.raw_html else None,
        )

        # Update webpage content
        if results:
            final_result = results[-1]
            webpage.cleaned_content = final_result.content_after
            if final_result.html_after:
                webpage.raw_html = final_result.html_after

        # Save logs
        await self._save_optimization_log(webpage, queries, results)

        return webpage, results

    async def _save_optimization_log(
        self,
        webpage: WebPage,
        queries: List[str],
        results: List[OptimizationResultV2],
    ) -> None:
        """Save optimization logs"""
        if not self._optimization_log_dir:
            return

        self._optimization_log_dir.mkdir(parents=True, exist_ok=True)

        safe_filename = (
            webpage.url.replace("https://", "")
            .replace("http://", "")
            .replace("/", "_")[:50]
        )
        output_file = self._optimization_log_dir / f"{self._output_prefix}{safe_filename}_v2.json"

        # Aggregate diagnosis statistics
        overall_diagnosis_stats: Dict[str, int] = {}
        for r in results:
            for cause, count in r.diagnosis_stats.items():
                overall_diagnosis_stats[cause] = overall_diagnosis_stats.get(cause, 0) + count

        log_data = {
            "run_id": self._run_id,
            "url": webpage.url,
            "version": "v2",
            "config": {
                "batch_size": self.batch_config.batch_size,
                "max_retries_per_query": self.batch_config.max_retries_per_query,
                "chunks_per_orchestra": self.batch_config.chunks_per_orchestra,
                "suggestion_merge_strategy": self.batch_config.suggestion_merge_strategy,
                "use_two_phase_analysis": self.batch_config.use_two_phase_analysis,
                "enable_policy_injection": self.batch_config.enable_policy_injection,
            },
            "queries": queries,
            "batches": [
                {
                    "batch_id": r.batch_id,
                    "queries": r.queries,
                    "suggestions_count": len(r.all_suggestions),
                    "applied_count": len(r.applied_modifications),
                    "success_rate_before": r.success_rate_before,
                    "diagnosis_stats": r.diagnosis_stats,
                }
                for r in results
            ],
            "overall_diagnosis_stats": overall_diagnosis_stats,
            "final_content": webpage.cleaned_content,
        }

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            print(f"Saved optimization log to {output_file}")
        except Exception as e:
            logger.warning(f"Failed to save optimization log: {e}")

    def optimize_page_batch(
        self,
        webpage: WebPage,
        queries: List[str],
    ) -> Tuple[WebPage, List[OptimizationResultV2]]:
        """Sync wrapper"""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.optimize_page_batch_async(webpage, queries))
        raise RuntimeError(
            "optimize_page_batch() cannot run inside an event loop; use await optimize_page_batch_async()."
        )

    async def evaluate_page_async(
        self,
        webpage: WebPage,
        queries: List[str],
        concurrency: Optional[int] = None,
    ) -> Dict[str, object]:
        """Evaluate page"""
        results: Dict[str, bool] = {}
        concurrency = concurrency or self.batch_config.max_concurrency
        semaphore = asyncio.Semaphore(concurrency)
        
        # Ensure queries is a Python list
        if hasattr(queries, 'tolist'):  # Check if numpy array
            queries = queries.tolist()
        elif not isinstance(queries, list):
            queries = list(queries)

        async def evaluate_one(query: str) -> Tuple[str, bool]:
            async with semaphore:
                retrieved_docs = await self._get_search_results(query)
                retrieved_docs, competitor_contents = await self._get_all_competitor_contents(
                    retrieved_docs
                )
                if not competitor_contents:
                    return query, False

                citation_result = await self.generator.generate_and_check(
                    query, webpage, retrieved_docs, competitor_contents
                )
                return query, citation_result.is_cited

        tasks = [asyncio.create_task(evaluate_one(query)) for query in queries]
        for coro in asyncio.as_completed(tasks):
            query, cited = await coro
            results[query] = cited

        # Safely compute ratio, avoiding numpy array issues
        if not queries:
            ratio = 0.0
        else:
            success_count = sum(results.values())
            queries_len = len(queries)
            ratio = float(success_count) / queries_len if queries_len > 0 else 0.0
        
        results["ratio"] = ratio
        return results


# Convenience functions
async def run_batch_geo_v2(
    url: str,
    content: str,
    queries: List[str],
    config_path: str = "geo_agent/config.yaml",
    batch_config: Optional[AgentGEOConfigV2] = None,
) -> Tuple[str, List[OptimizationResultV2]]:
    """
    Run Batch GEO V2 optimization

    Args:
        url: Webpage URL
        content: Webpage content
        queries: List of queries
        config_path: Config file path
        batch_config: Batch processing config

    Returns:
        Tuple[str, List[OptimizationResultV2]]: (optimized content, batch results list)
    """
    webpage = WebPage(url=url, raw_html="", cleaned_content=content)
    agent = AgentGEOV2(config_path=config_path, batch_config=batch_config)
    optimized_page, results = await agent.optimize_page_batch_async(webpage, queries)
    return optimized_page.cleaned_content, results

"""
Multi-Document Optimizer

Implements document (WebPage) level parallel processing capability with control over concurrent document count.

Architecture:
```
MultiDocumentOptimizer (top-level control)
├── doc_semaphore: controls document-level concurrency
├── shared resource pool: llm, search_engine, generator, cache
└── optimize_documents_async()
        └── parallel scheduling of multiple AgentGEOV2
```

Key Features:
1. Document-level parallelism: control concurrent documents via semaphore
2. Resource sharing: LLM, search engine, cache shared across all documents
3. Error isolation: single document failure doesn't affect others (configurable fail_fast)
4. Progress callback: supports real-time progress notification
5. Flexible query assignment: supports per-document or shared queries
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from geo_agent.config import get_llm_from_config, load_config
from geo_agent.core.models import SearchResult, WebPage
from geo_agent.search_engine.chatnoir import ChatNoirClient
from geo_agent.search_engine.manager import SearchManager
from geo_agent.utils.html_parser import HtmlParser

from .agent_geo import AsyncInContextGeneratorV2, REPO_ROOT
from .suggestion_processor import SuggestionProcessorV2
from .memory_manager import HistoryManagerV2
from .models import (
    AgentGEOConfigV2,
    OptimizationResultV2,
    DocumentOptimizationResult,
    MultiDocConfigV2,
    MultiDocOptimizationResult,
)

logger = logging.getLogger(__name__)


# Progress callback type
ProgressCallback = Callable[[int, int, str, Optional[str]], None]
# Signature: (completed_count, total_count, current_url, status_message)


class MultiDocumentOptimizer:
    """
    Multi-document parallel optimizer

    Optimizes multiple WebPages simultaneously, controls document-level concurrency, shares resources for efficiency.

    Example:
        ```python
        from batch_suggestion_orchestrator import MultiDocumentOptimizer, MultiDocConfigV2

        config = MultiDocConfigV2(
            max_doc_concurrency=3,
            batch_config=AgentGEOConfigV2(batch_size=10, max_concurrency=4),
        )
        optimizer = MultiDocumentOptimizer(multi_config=config)

        result = await optimizer.optimize_documents_with_shared_queries(
            webpages=[page1, page2, page3],
            shared_queries=["query1", "query2", ...],
        )
        print(f"Success: {result.success_count}/{result.total_count}")
        ```
    """

    def __init__(
        self,
        config_path: str = "geo_agent/config.yaml",
        multi_config: Optional[MultiDocConfigV2] = None,
        disk_cache_dir: Optional[str] = None,
        optimization_log_dir: Optional[str] = None,
        output_prefix: str = "",
    ):
        """
        Initialize multi-document optimizer

        Args:
            config_path: Config file path
            multi_config: Multi-document config, default MultiDocConfigV2()
            disk_cache_dir: Disk cache directory (shared across documents)
            optimization_log_dir: Optimization log directory
            output_prefix: Output file prefix
        """
        # Config path
        config_file = Path(config_path)
        if not config_file.is_absolute():
            config_file = (REPO_ROOT / config_file).resolve()
        self.config_path = str(config_file)

        # Multi-document config
        self.multi_config = multi_config or MultiDocConfigV2()

        # Initialize shared resources
        self.llm = get_llm_from_config(self.config_path)
        self.search_engine = SearchManager(self.config_path)
        self.generator = AsyncInContextGeneratorV2(self.config_path)

        # Search engine type
        config = load_config(self.config_path)
        search_config = config.get("search", {})
        self._search_provider = search_config.get("provider", "chatnoir")

        if self._search_provider == "chatnoir":
            self._chatnoir_client = ChatNoirClient()
        else:
            self._chatnoir_client = None

        self._html_parser = HtmlParser(self.config_path)

        # Shared cache
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

        # Document-level semaphore (controls concurrency)
        self._doc_semaphore: Optional[asyncio.Semaphore] = None

        # Shared history manager
        history_path = None
        if self.multi_config.batch_config.history_persist_path:
            history_path = Path(self.multi_config.batch_config.history_persist_path)
        self._shared_history_manager = HistoryManagerV2(persist_path=history_path)

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Get or create semaphore (need new one for each event loop)"""
        if self._doc_semaphore is None:
            self._doc_semaphore = asyncio.Semaphore(
                self.multi_config.max_doc_concurrency
            )
        return self._doc_semaphore

    async def _get_search_results(self, query: str) -> List[SearchResult]:
        """Get search results (with shared cache)"""
        import hashlib
        import json

        async with self._cache_lock:
            cached = self._search_cache.get(query)
        if cached is not None:
            return cached

        # Check disk cache
        cache_key = hashlib.sha256(query.encode("utf-8")).hexdigest()
        cache_path = self._disk_cache_dir / f"{cache_key}.json" if self._disk_cache_dir else None

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

        # Execute search
        results = await asyncio.to_thread(self.search_engine.search, query)
        async with self._cache_lock:
            self._search_cache[query] = results

        # Write to disk cache
        if cache_path:
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(
                        [r.model_dump() if hasattr(r, "model_dump") else r.dict() for r in results],
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
            except Exception as exc:
                logger.warning("Failed to write disk cache: %s", exc)

        return results

    async def _get_competitor_full_content(self, doc: SearchResult) -> Optional[str]:
        """Get competitor full content (shared cache)"""
        if not doc.uuid:
            return doc.snippet or None

        if doc.uuid in self._html_content_cache:
            return self._html_content_cache[doc.uuid]

        if self._search_provider == "tavily":
            if doc.raw_content:
                self._html_content_cache[doc.uuid] = doc.raw_content
                return doc.raw_content
            return doc.snippet or None
        else:
            try:
                html_content = await asyncio.to_thread(
                    self._chatnoir_client.get_html_content, doc.uuid
                )
                if html_content:
                    parsed_content = self._html_parser.parse(html_content)
                    self._html_content_cache[doc.uuid] = parsed_content
                    return parsed_content
            except Exception as e:
                logger.warning("ChatNoir fetch failed for %s: %s", doc.uuid, e)
            return doc.snippet or None

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

    async def _process_single_document(
        self,
        webpage: WebPage,
        queries: List[str],
        doc_index: int,
        total_docs: int,
        on_progress: Optional[ProgressCallback] = None,
    ) -> DocumentOptimizationResult:
        """
        Process single document (with semaphore control)

        Args:
            webpage: WebPage to optimize
            queries: Query list for this document
            doc_index: Document index
            total_docs: Total document count
            on_progress: Progress callback

        Returns:
            DocumentOptimizationResult
        """
        semaphore = self._get_semaphore()

        async with semaphore:
            start_time = time.time()
            url_short = webpage.url[:50] if webpage.url else f"doc_{doc_index}"

            logger.info(f"[{doc_index + 1}/{total_docs}] Starting: {url_short}")
            if on_progress:
                on_progress(doc_index, total_docs, webpage.url, "starting")

            try:
                # Create batch processor (inject shared resources)
                processor = SuggestionProcessorV2(
                    llm=self.llm,
                    generator=self.generator,
                    config=self.multi_config.batch_config,
                    history_manager=self._shared_history_manager,
                    search_func=self._get_search_results,
                    competitor_content_func=self._get_all_competitor_contents,
                )

                # Execute optimization
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

                duration_ms = (time.time() - start_time) * 1000
                suggestions_applied = sum(len(r.applied_modifications) for r in results)

                logger.info(
                    f"[{doc_index + 1}/{total_docs}] Completed: {url_short} "
                    f"({duration_ms:.0f}ms, {suggestions_applied} suggestions applied)"
                )

                if on_progress:
                    on_progress(doc_index + 1, total_docs, webpage.url, "completed")

                return DocumentOptimizationResult(
                    webpage=webpage,
                    batch_results=results,
                    success=True,
                    error=None,
                    duration_ms=duration_ms,
                    queries_count=len(queries),
                    suggestions_applied=suggestions_applied,
                )

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                error_msg = str(e)
                logger.error(f"[{doc_index + 1}/{total_docs}] Failed: {url_short} - {error_msg}")

                if on_progress:
                    on_progress(doc_index + 1, total_docs, webpage.url, f"failed: {error_msg[:50]}")

                return DocumentOptimizationResult(
                    webpage=webpage,
                    batch_results=[],
                    success=False,
                    error=error_msg,
                    duration_ms=duration_ms,
                    queries_count=len(queries),
                    suggestions_applied=0,
                )

    async def _process_document_with_retry(
        self,
        webpage: WebPage,
        queries: List[str],
        doc_index: int,
        total_docs: int,
        on_progress: Optional[ProgressCallback] = None,
    ) -> DocumentOptimizationResult:
        """
        Document processing with retry

        Args:
            webpage: WebPage to optimize
            queries: Query list
            doc_index: Document index
            total_docs: Total document count
            on_progress: Progress callback

        Returns:
            DocumentOptimizationResult
        """
        max_retries = self.multi_config.max_retries_per_doc
        last_result: Optional[DocumentOptimizationResult] = None

        for attempt in range(max_retries + 1):
            if attempt > 0:
                logger.info(
                    f"[{doc_index + 1}/{total_docs}] Retry {attempt}/{max_retries}: {webpage.url[:50]}"
                )

            result = await self._process_single_document(
                webpage=webpage,
                queries=queries,
                doc_index=doc_index,
                total_docs=total_docs,
                on_progress=on_progress if attempt == 0 else None,  # Only report progress on first attempt
            )

            if result.success:
                return result

            last_result = result

            # Brief delay before retry
            if attempt < max_retries:
                await asyncio.sleep(1.0 * (attempt + 1))

        return last_result or DocumentOptimizationResult(
            webpage=webpage,
            batch_results=[],
            success=False,
            error="Max retries exceeded",
            duration_ms=0,
            queries_count=len(queries),
            suggestions_applied=0,
        )

    async def optimize_documents_async(
        self,
        webpages: List[WebPage],
        queries_per_page: List[List[str]],
        on_progress: Optional[ProgressCallback] = None,
    ) -> MultiDocOptimizationResult:
        """
        Parallel optimization of multiple documents (each with independent queries)

        Args:
            webpages: WebPage list
            queries_per_page: Queries list for each document, length must match webpages
            on_progress: Progress callback (completed, total, url, status)

        Returns:
            MultiDocOptimizationResult

        Raises:
            ValueError: If webpages and queries_per_page lengths don't match
        """
        if len(webpages) != len(queries_per_page):
            raise ValueError(
                f"webpages ({len(webpages)}) and queries_per_page ({len(queries_per_page)}) "
                "must have the same length"
            )

        if not webpages:
            return MultiDocOptimizationResult(results=[], total_duration_ms=0)

        # Reset semaphore (new event loop)
        self._doc_semaphore = asyncio.Semaphore(self.multi_config.max_doc_concurrency)

        start_time = time.time()
        total_docs = len(webpages)

        logger.info(
            f"Starting multi-document optimization: {total_docs} documents, "
            f"max_concurrency={self.multi_config.max_doc_concurrency}"
        )

        # Create all tasks
        tasks = [
            self._process_document_with_retry(
                webpage=wp,
                queries=queries,
                doc_index=idx,
                total_docs=total_docs,
                on_progress=on_progress,
            )
            for idx, (wp, queries) in enumerate(zip(webpages, queries_per_page))
        ]

        # Concurrent execution
        if self.multi_config.fail_fast:
            # fail_fast: cancel all if any fails
            results = []
            for coro in asyncio.as_completed(tasks):
                try:
                    result = await coro
                    if not result.success:
                        # Cancel remaining tasks
                        for task in tasks:
                            if isinstance(task, asyncio.Task) and not task.done():
                                task.cancel()
                        raise RuntimeError(f"Document failed: {result.error}")
                    results.append(result)
                except asyncio.CancelledError:
                    break
        else:
            # Default: collect all results (including failures)
            results = await asyncio.gather(*tasks, return_exceptions=False)

        total_duration_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Multi-document optimization completed in {total_duration_ms:.0f}ms"
        )

        return MultiDocOptimizationResult(
            results=results,
            total_duration_ms=total_duration_ms,
        )

    async def optimize_documents_with_shared_queries(
        self,
        webpages: List[WebPage],
        shared_queries: List[str],
        on_progress: Optional[ProgressCallback] = None,
    ) -> MultiDocOptimizationResult:
        """
        Parallel optimization of multiple documents (all share same queries)

        This is the more common use case: one set of queries for testing/optimizing multiple target documents.

        Args:
            webpages: WebPage list
            shared_queries: Queries shared by all documents
            on_progress: Progress callback

        Returns:
            MultiDocOptimizationResult
        """
        queries_per_page = [shared_queries for _ in webpages]
        return await self.optimize_documents_async(
            webpages=webpages,
            queries_per_page=queries_per_page,
            on_progress=on_progress,
        )

    def optimize_documents(
        self,
        webpages: List[WebPage],
        queries_per_page: List[List[str]],
        on_progress: Optional[ProgressCallback] = None,
    ) -> MultiDocOptimizationResult:
        """
        Sync wrapper: parallel optimization of multiple documents

        Args:
            webpages: WebPage list
            queries_per_page: Queries list for each document
            on_progress: Progress callback

        Returns:
            MultiDocOptimizationResult
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.optimize_documents_async(webpages, queries_per_page, on_progress)
            )
        raise RuntimeError(
            "optimize_documents() cannot run inside an event loop; "
            "use await optimize_documents_async()."
        )

    def optimize_documents_shared(
        self,
        webpages: List[WebPage],
        shared_queries: List[str],
        on_progress: Optional[ProgressCallback] = None,
    ) -> MultiDocOptimizationResult:
        """
        Sync wrapper: optimize multiple documents with shared queries

        Args:
            webpages: WebPage list
            shared_queries: Shared queries
            on_progress: Progress callback

        Returns:
            MultiDocOptimizationResult
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.optimize_documents_with_shared_queries(
                    webpages, shared_queries, on_progress
                )
            )
        raise RuntimeError(
            "optimize_documents_shared() cannot run inside an event loop; "
            "use await optimize_documents_with_shared_queries()."
        )

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            "search_cache_size": len(self._search_cache),
            "html_cache_size": len(self._html_content_cache),
        }

    def clear_cache(self) -> None:
        """Clear all cache"""
        self._search_cache.clear()
        self._html_content_cache.clear()
        logger.info("Cache cleared")


# Convenience functions
async def optimize_multiple_documents(
    webpages: List[WebPage],
    queries: List[str],
    config_path: str = "geo_agent/config.yaml",
    max_doc_concurrency: int = 2,
    disk_cache_dir: Optional[str] = None,
) -> MultiDocOptimizationResult:
    """
    Convenience function: optimize multiple documents

    Args:
        webpages: WebPage list
        queries: Shared queries
        config_path: Config file path
        max_doc_concurrency: Document concurrency
        disk_cache_dir: Disk cache directory

    Returns:
        MultiDocOptimizationResult
    """
    config = MultiDocConfigV2(max_doc_concurrency=max_doc_concurrency)
    optimizer = MultiDocumentOptimizer(
        config_path=config_path,
        multi_config=config,
        disk_cache_dir=disk_cache_dir,
    )
    return await optimizer.optimize_documents_with_shared_queries(webpages, queries)

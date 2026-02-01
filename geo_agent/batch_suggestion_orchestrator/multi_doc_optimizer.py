"""
多文档并行优化器 (Multi-Document Optimizer)

实现文档（WebPage）级别的并行处理能力，可控制同时优化的文档数量。

架构设计：
```
MultiDocumentOptimizer (顶层控制)
├── doc_semaphore: 控制文档级并发
├── 共享资源池: llm, search_engine, generator, 缓存
└── optimize_documents_async()
        └── 并行调度多个 AgentGEOV2
```

主要特性：
1. 文档级并行：通过 semaphore 控制同时处理的文档数
2. 资源共享：LLM、搜索引擎、缓存在所有文档间共享
3. 错误隔离：单个文档失败不影响其他文档（可配置 fail_fast）
4. 进度回调：支持实时进度通知
5. 灵活的 query 分配：支持每文档独立 queries 或共享 queries
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


# 进度回调类型
ProgressCallback = Callable[[int, int, str, Optional[str]], None]
# 签名: (completed_count, total_count, current_url, status_message)


class MultiDocumentOptimizer:
    """
    多文档并行优化器

    用于同时优化多个 WebPage，控制文档级并发数量，共享资源以提高效率。

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
        print(f"成功: {result.success_count}/{result.total_count}")
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
        初始化多文档优化器

        Args:
            config_path: 配置文件路径
            multi_config: 多文档配置，默认 MultiDocConfigV2()
            disk_cache_dir: 磁盘缓存目录（跨文档共享）
            optimization_log_dir: 优化日志目录
            output_prefix: 输出文件前缀
        """
        # 配置路径
        config_file = Path(config_path)
        if not config_file.is_absolute():
            config_file = (REPO_ROOT / config_file).resolve()
        self.config_path = str(config_file)

        # 多文档配置
        self.multi_config = multi_config or MultiDocConfigV2()

        # 共享资源初始化
        self.llm = get_llm_from_config(self.config_path)
        self.search_engine = SearchManager(self.config_path)
        self.generator = AsyncInContextGeneratorV2(self.config_path)

        # 搜索引擎类型
        config = load_config(self.config_path)
        search_config = config.get("search", {})
        self._search_provider = search_config.get("provider", "chatnoir")

        if self._search_provider == "chatnoir":
            self._chatnoir_client = ChatNoirClient()
        else:
            self._chatnoir_client = None

        self._html_parser = HtmlParser(self.config_path)

        # 共享缓存
        self._html_content_cache: Dict[str, str] = {}
        self._search_cache: Dict[str, List[SearchResult]] = {}
        self._cache_lock = asyncio.Lock()

        # 磁盘缓存
        self._disk_cache_dir: Optional[Path] = None
        if disk_cache_dir:
            cache_path = Path(disk_cache_dir)
            if not cache_path.is_absolute():
                cache_path = (REPO_ROOT / cache_path).resolve()
            cache_path.mkdir(parents=True, exist_ok=True)
            self._disk_cache_dir = cache_path

        # 日志目录
        self._optimization_log_dir: Optional[Path] = None
        if optimization_log_dir:
            log_path = Path(optimization_log_dir)
            if not log_path.is_absolute():
                log_path = (REPO_ROOT / log_path).resolve()
            self._optimization_log_dir = log_path

        self._output_prefix = output_prefix

        # 文档级信号量（控制并发）
        self._doc_semaphore: Optional[asyncio.Semaphore] = None

        # 共享历史管理器
        history_path = None
        if self.multi_config.batch_config.history_persist_path:
            history_path = Path(self.multi_config.batch_config.history_persist_path)
        self._shared_history_manager = HistoryManagerV2(persist_path=history_path)

    def _get_semaphore(self) -> asyncio.Semaphore:
        """获取或创建信号量（每个事件循环需要新建）"""
        if self._doc_semaphore is None:
            self._doc_semaphore = asyncio.Semaphore(
                self.multi_config.max_doc_concurrency
            )
        return self._doc_semaphore

    async def _get_search_results(self, query: str) -> List[SearchResult]:
        """获取搜索结果（带共享缓存）"""
        import hashlib
        import json

        async with self._cache_lock:
            cached = self._search_cache.get(query)
        if cached is not None:
            return cached

        # 检查磁盘缓存
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

        # 执行搜索
        results = await asyncio.to_thread(self.search_engine.search, query)
        async with self._cache_lock:
            self._search_cache[query] = results

        # 写入磁盘缓存
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
        """获取竞争对手完整内容（共享缓存）"""
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
        """获取所有竞争对手内容"""
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
        处理单个文档（带信号量控制）

        Args:
            webpage: 要优化的网页
            queries: 该文档的查询列表
            doc_index: 文档索引
            total_docs: 总文档数
            on_progress: 进度回调

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
                # 创建批处理器（注入共享资源）
                processor = SuggestionProcessorV2(
                    llm=self.llm,
                    generator=self.generator,
                    config=self.multi_config.batch_config,
                    history_manager=self._shared_history_manager,
                    search_func=self._get_search_results,
                    competitor_content_func=self._get_all_competitor_contents,
                )

                # 执行优化
                results = await processor.process_all_batches(
                    content=webpage.cleaned_content,
                    all_queries=queries,
                    raw_html=webpage.raw_html if webpage.raw_html else None,
                )

                # 更新网页内容
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
        带重试的文档处理

        Args:
            webpage: 要优化的网页
            queries: 查询列表
            doc_index: 文档索引
            total_docs: 总文档数
            on_progress: 进度回调

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
                on_progress=on_progress if attempt == 0 else None,  # 只在首次报告进度
            )

            if result.success:
                return result

            last_result = result

            # 短暂延迟后重试
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
        并行优化多个文档（每个文档有独立的 queries）

        Args:
            webpages: WebPage 列表
            queries_per_page: 每个文档对应的 queries 列表，长度必须与 webpages 相同
            on_progress: 进度回调 (completed, total, url, status)

        Returns:
            MultiDocOptimizationResult

        Raises:
            ValueError: 如果 webpages 和 queries_per_page 长度不匹配
        """
        if len(webpages) != len(queries_per_page):
            raise ValueError(
                f"webpages ({len(webpages)}) and queries_per_page ({len(queries_per_page)}) "
                "must have the same length"
            )

        if not webpages:
            return MultiDocOptimizationResult(results=[], total_duration_ms=0)

        # 重置信号量（新的事件循环）
        self._doc_semaphore = asyncio.Semaphore(self.multi_config.max_doc_concurrency)

        start_time = time.time()
        total_docs = len(webpages)

        logger.info(
            f"Starting multi-document optimization: {total_docs} documents, "
            f"max_concurrency={self.multi_config.max_doc_concurrency}"
        )

        # 创建所有任务
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

        # 并发执行
        if self.multi_config.fail_fast:
            # fail_fast: 任一失败则取消全部
            results = []
            for coro in asyncio.as_completed(tasks):
                try:
                    result = await coro
                    if not result.success:
                        # 取消剩余任务
                        for task in tasks:
                            if isinstance(task, asyncio.Task) and not task.done():
                                task.cancel()
                        raise RuntimeError(f"Document failed: {result.error}")
                    results.append(result)
                except asyncio.CancelledError:
                    break
        else:
            # 默认：收集所有结果（包括失败）
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
        并行优化多个文档（所有文档共用同一组 queries）

        这是更常见的使用场景：一组 queries 用于测试/优化多个目标文档。

        Args:
            webpages: WebPage 列表
            shared_queries: 所有文档共用的 queries
            on_progress: 进度回调

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
        同步包装器：并行优化多个文档

        Args:
            webpages: WebPage 列表
            queries_per_page: 每个文档对应的 queries 列表
            on_progress: 进度回调

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
        同步包装器：使用共享 queries 优化多个文档

        Args:
            webpages: WebPage 列表
            shared_queries: 共享的 queries
            on_progress: 进度回调

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
        """获取缓存统计"""
        return {
            "search_cache_size": len(self._search_cache),
            "html_cache_size": len(self._html_content_cache),
        }

    def clear_cache(self) -> None:
        """清空所有缓存"""
        self._search_cache.clear()
        self._html_content_cache.clear()
        logger.info("Cache cleared")


# 便捷函数
async def optimize_multiple_documents(
    webpages: List[WebPage],
    queries: List[str],
    config_path: str = "geo_agent/config.yaml",
    max_doc_concurrency: int = 2,
    disk_cache_dir: Optional[str] = None,
) -> MultiDocOptimizationResult:
    """
    便捷函数：优化多个文档

    Args:
        webpages: WebPage 列表
        queries: 共享的 queries
        config_path: 配置文件路径
        max_doc_concurrency: 文档并发数
        disk_cache_dir: 磁盘缓存目录

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

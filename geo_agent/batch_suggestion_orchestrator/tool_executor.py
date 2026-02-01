"""
Batch GEO V2 工具执行器
异步执行 geo_agent 工具，支持 Batch 模式的上下文注入
"""
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 设置路径
REPO_ROOT = Path(__file__).resolve().parents[1]
GEO_AGENT_ROOT = REPO_ROOT / "geo_agent"
if str(GEO_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(GEO_AGENT_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

from geo_agent.core.memory import parse_tool_output
from geo_agent.tools import registry

logger = logging.getLogger(__name__)


class ToolExecutor:
    """
    异步工具执行器

    负责：
    1. 注入上下文参数（target_content, context_before, context_after 等）
    2. 异步执行工具
    3. 解析工具输出
    """

    def __init__(self):
        self.tools_map = registry.tools

    def get_tool(self, tool_name: str):
        """获取工具实例"""
        return registry.get_tool(tool_name)

    def get_all_tools(self):
        """获取所有工具"""
        return registry.get_all_tools()

    async def execute_tool_async(
        self,
        tool_name: str,
        tool_arguments: Dict[str, Any],
        target_content: str,
        context_before: str = "",
        context_after: str = "",
        core_idea: str = "",
        previous_modifications: str = "",
    ) -> Tuple[str, List[str]]:
        """
        异步执行工具

        Args:
            tool_name: 工具名称
            tool_arguments: 工具参数（来自 LLM）
            target_content: 目标内容
            context_before: 前文上下文
            context_after: 后文上下文
            core_idea: 核心思想
            previous_modifications: 之前的修改记录

        Returns:
            Tuple[str, List[str]]: (修改后的内容, 关键改动列表)
        """
        tool = self.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found in registry")

        # 注入上下文参数
        args = tool_arguments.copy()
        args["target_content"] = target_content
        args["context_before"] = context_before
        args["context_after"] = context_after
        args["core_idea"] = core_idea
        args["previous_modifications"] = previous_modifications

        # 异步执行工具
        try:
            raw_output = await asyncio.to_thread(tool.run, args)
        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name}, error: {e}")
            raise RuntimeError(f"Tool execution failed: {e}") from e

        # 解析输出
        modified_content, key_changes = parse_tool_output(raw_output)

        return modified_content, key_changes

    async def execute_with_context_injection(
        self,
        tool_name: str,
        tool_arguments: Dict[str, Any],
        chunks: List[Any],  # ContentChunk 列表
        target_chunk_index: int,
        core_idea: str = "",
        previous_modifications: str = "",
    ) -> Tuple[str, List[str], int]:
        """
        执行工具并自动注入上下文

        从 chunks 列表中提取目标内容和上下文

        Args:
            tool_name: 工具名称
            tool_arguments: 工具参数
            chunks: ContentChunk 列表
            target_chunk_index: 目标 chunk 索引
            core_idea: 核心思想
            previous_modifications: 之前的修改记录

        Returns:
            Tuple[str, List[str], int]: (修改后的内容, 关键改动列表, 目标索引)
        """
        if not chunks:
            raise ValueError("Empty chunks list")

        # 边界检查
        target_idx = max(0, min(target_chunk_index, len(chunks) - 1))

        # 提取目标内容
        target_chunk = chunks[target_idx]
        target_content = target_chunk.html if hasattr(target_chunk, "html") else target_chunk.text

        # 提取上下文
        context_before = ""
        context_after = ""

        if target_idx > 0:
            prev_chunk = chunks[target_idx - 1]
            context_before = prev_chunk.text if hasattr(prev_chunk, "text") else str(prev_chunk)

        if target_idx < len(chunks) - 1:
            next_chunk = chunks[target_idx + 1]
            context_after = next_chunk.text if hasattr(next_chunk, "text") else str(next_chunk)

        # 执行工具
        modified_content, key_changes = await self.execute_tool_async(
            tool_name=tool_name,
            tool_arguments=tool_arguments,
            target_content=target_content,
            context_before=context_before,
            context_after=context_after,
            core_idea=core_idea,
            previous_modifications=previous_modifications,
        )

        return modified_content, key_changes, target_idx


class BatchToolExecutor(ToolExecutor):
    """
    批量工具执行器

    支持：
    1. 并行执行多个工具调用
    2. 冲突检测和解决
    3. 结果聚合
    """

    def __init__(self, max_concurrency: int = 4):
        super().__init__()
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def execute_batch_async(
        self,
        tool_calls: List[Dict[str, Any]],
        chunks: List[Any],
        core_idea: str = "",
        previous_modifications: str = "",
    ) -> List[Tuple[str, List[str], int, Optional[Exception]]]:
        """
        批量执行工具调用

        Args:
            tool_calls: 工具调用列表，每个元素包含：
                - tool_name: str
                - tool_arguments: Dict
                - target_chunk_index: int
            chunks: ContentChunk 列表
            core_idea: 核心思想
            previous_modifications: 之前的修改记录

        Returns:
            List[Tuple[str, List[str], int, Optional[Exception]]]:
                每个元素为 (修改后内容, 关键改动, 目标索引, 异常或 None)
        """

        async def execute_one(call: Dict[str, Any]) -> Tuple[str, List[str], int, Optional[Exception]]:
            async with self.semaphore:
                try:
                    modified, changes, idx = await self.execute_with_context_injection(
                        tool_name=call["tool_name"],
                        tool_arguments=call.get("tool_arguments", {}),
                        chunks=chunks,
                        target_chunk_index=call.get("target_chunk_index", 0),
                        core_idea=core_idea,
                        previous_modifications=previous_modifications,
                    )
                    return modified, changes, idx, None
                except Exception as e:
                    logger.error(f"Batch tool execution failed: {call.get('tool_name')}, error: {e}")
                    return "", [], call.get("target_chunk_index", 0), e

        tasks = [asyncio.create_task(execute_one(call)) for call in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        return results

    def detect_conflicts(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """
        检测冲突的工具调用（同一 chunk 的多个修改）

        Args:
            tool_calls: 工具调用列表

        Returns:
            List[List[Dict]]: 冲突组列表
        """
        # 按 target_chunk_index 分组
        by_chunk: Dict[int, List[Dict[str, Any]]] = {}
        for call in tool_calls:
            idx = call.get("target_chunk_index", 0)
            if idx not in by_chunk:
                by_chunk[idx] = []
            by_chunk[idx].append(call)

        # 返回有冲突的组（同一 chunk 有多个调用）
        conflicts = [calls for calls in by_chunk.values() if len(calls) > 1]

        return conflicts

    async def execute_with_conflict_resolution(
        self,
        tool_calls: List[Dict[str, Any]],
        chunks: List[Any],
        core_idea: str = "",
        previous_modifications: str = "",
        resolution_strategy: str = "priority",
    ) -> Dict[int, Tuple[str, List[str]]]:
        """
        执行工具调用并解决冲突

        Args:
            tool_calls: 工具调用列表
            chunks: ContentChunk 列表
            core_idea: 核心思想
            previous_modifications: 之前的修改记录
            resolution_strategy: 冲突解决策略
                - "priority": 使用优先级最高的调用
                - "first": 使用第一个调用
                - "merge": 尝试合并（暂未实现）

        Returns:
            Dict[int, Tuple[str, List[str]]]: chunk_index -> (修改内容, 关键改动)
        """
        # 检测冲突
        conflicts = self.detect_conflicts(tool_calls)

        if conflicts:
            logger.warning(f"Detected {len(conflicts)} conflict groups")

        # 解决冲突：选择每个 chunk 的最佳调用
        resolved_calls: Dict[int, Dict[str, Any]] = {}

        for call in tool_calls:
            idx = call.get("target_chunk_index", 0)

            if idx not in resolved_calls:
                resolved_calls[idx] = call
            else:
                # 冲突解决
                if resolution_strategy == "priority":
                    # 比较优先级分数
                    existing_priority = call.get("priority_score", 0.5)
                    new_priority = call.get("priority_score", 0.5)
                    if new_priority > existing_priority:
                        resolved_calls[idx] = call
                elif resolution_strategy == "first":
                    # 保持第一个
                    pass

        # 执行解决后的调用
        final_calls = list(resolved_calls.values())
        results = await self.execute_batch_async(
            final_calls, chunks, core_idea, previous_modifications
        )

        # 组织结果
        output: Dict[int, Tuple[str, List[str]]] = {}
        for modified, changes, idx, error in results:
            if error is None:
                output[idx] = (modified, changes)
            else:
                logger.warning(f"Tool execution failed for chunk {idx}: {error}")

        return output

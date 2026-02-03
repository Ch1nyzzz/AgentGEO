"""
Batch GEO V2 Tool Executor
Async execution of geo_agent tools with Batch mode context injection support
"""
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Setup paths
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
    Async Tool Executor

    Responsible for:
    1. Injecting context parameters (target_content, context_before, context_after, etc.)
    2. Async tool execution
    3. Parsing tool output
    """

    def __init__(self):
        self.tools_map = registry.tools

    def get_tool(self, tool_name: str):
        """Get tool instance"""
        return registry.get_tool(tool_name)

    def get_all_tools(self):
        """Get all tools"""
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
        Async tool execution

        Args:
            tool_name: Tool name
            tool_arguments: Tool arguments (from LLM)
            target_content: Target content
            context_before: Context before target
            context_after: Context after target
            core_idea: Core idea
            previous_modifications: Previous modification records

        Returns:
            Tuple[str, List[str]]: (Modified content, list of key changes)
        """
        tool = self.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found in registry")

        # Inject context parameters
        args = tool_arguments.copy()
        args["target_content"] = target_content
        args["context_before"] = context_before
        args["context_after"] = context_after
        args["core_idea"] = core_idea
        args["previous_modifications"] = previous_modifications

        # Async tool execution
        try:
            raw_output = await asyncio.to_thread(tool.run, args)
        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name}, error: {e}")
            raise RuntimeError(f"Tool execution failed: {e}") from e

        # Parse output
        modified_content, key_changes = parse_tool_output(raw_output)

        return modified_content, key_changes

    async def execute_with_context_injection(
        self,
        tool_name: str,
        tool_arguments: Dict[str, Any],
        chunks: List[Any],  # ContentChunk list
        target_chunk_index: int,
        core_idea: str = "",
        previous_modifications: str = "",
    ) -> Tuple[str, List[str], int]:
        """
        Execute tool with automatic context injection

        Extracts target content and context from chunks list

        Args:
            tool_name: Tool name
            tool_arguments: Tool arguments
            chunks: ContentChunk list
            target_chunk_index: Target chunk index
            core_idea: Core idea
            previous_modifications: Previous modification records

        Returns:
            Tuple[str, List[str], int]: (Modified content, list of key changes, target index)
        """
        if not chunks:
            raise ValueError("Empty chunks list")

        # Boundary check
        target_idx = max(0, min(target_chunk_index, len(chunks) - 1))

        # Extract target content
        target_chunk = chunks[target_idx]
        target_content = target_chunk.html if hasattr(target_chunk, "html") else target_chunk.text

        # Extract context
        context_before = ""
        context_after = ""

        if target_idx > 0:
            prev_chunk = chunks[target_idx - 1]
            context_before = prev_chunk.text if hasattr(prev_chunk, "text") else str(prev_chunk)

        if target_idx < len(chunks) - 1:
            next_chunk = chunks[target_idx + 1]
            context_after = next_chunk.text if hasattr(next_chunk, "text") else str(next_chunk)

        # Execute tool
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
    Batch Tool Executor

    Supports:
    1. Parallel execution of multiple tool calls
    2. Conflict detection and resolution
    3. Result aggregation
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
        Batch execute tool calls

        Args:
            tool_calls: List of tool calls, each containing:
                - tool_name: str
                - tool_arguments: Dict
                - target_chunk_index: int
            chunks: ContentChunk list
            core_idea: Core idea
            previous_modifications: Previous modification records

        Returns:
            List[Tuple[str, List[str], int, Optional[Exception]]]:
                Each element is (modified content, key changes, target index, exception or None)
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
        Detect conflicting tool calls (multiple modifications to same chunk)

        Args:
            tool_calls: List of tool calls

        Returns:
            List[List[Dict]]: List of conflict groups
        """
        # Group by target_chunk_index
        by_chunk: Dict[int, List[Dict[str, Any]]] = {}
        for call in tool_calls:
            idx = call.get("target_chunk_index", 0)
            if idx not in by_chunk:
                by_chunk[idx] = []
            by_chunk[idx].append(call)

        # Return conflicting groups (same chunk has multiple calls)
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
        Execute tool calls and resolve conflicts

        Args:
            tool_calls: List of tool calls
            chunks: ContentChunk list
            core_idea: Core idea
            previous_modifications: Previous modification records
            resolution_strategy: Conflict resolution strategy
                - "priority": Use highest priority call
                - "first": Use first call
                - "merge": Try to merge (not yet implemented)

        Returns:
            Dict[int, Tuple[str, List[str]]]: chunk_index -> (modified content, key changes)
        """
        # Detect conflicts
        conflicts = self.detect_conflicts(tool_calls)

        if conflicts:
            logger.warning(f"Detected {len(conflicts)} conflict groups")

        # Resolve conflicts: select best call for each chunk
        resolved_calls: Dict[int, Dict[str, Any]] = {}

        for call in tool_calls:
            idx = call.get("target_chunk_index", 0)

            if idx not in resolved_calls:
                resolved_calls[idx] = call
            else:
                # Conflict resolution
                if resolution_strategy == "priority":
                    # Compare priority scores
                    existing_priority = call.get("priority_score", 0.5)
                    new_priority = call.get("priority_score", 0.5)
                    if new_priority > existing_priority:
                        resolved_calls[idx] = call
                elif resolution_strategy == "first":
                    # Keep the first one
                    pass

        # Execute resolved calls
        final_calls = list(resolved_calls.values())
        results = await self.execute_batch_async(
            final_calls, chunks, core_idea, previous_modifications
        )

        # Organize results
        output: Dict[int, Tuple[str, List[str]]] = {}
        for modified, changes, idx, error in results:
            if error is None:
                output[idx] = (modified, changes)
            else:
                logger.warning(f"Tool execution failed for chunk {idx}: {error}")

        return output

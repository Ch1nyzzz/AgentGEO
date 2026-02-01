"""
Batch GEO V2 建议收集器
使用两阶段分析（诊断 + 策略选择）收集优化建议

V2.1 更新：
- 实现与 GEO Agent 完全一致的 retry 验证机制
- 在临时结构上测试修改，检查是否生效
- 只返回最终生效的建议
"""
import asyncio
import logging
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# 设置路径
REPO_ROOT = Path(__file__).resolve().parents[1]
GEO_AGENT_ROOT = REPO_ROOT / "geo_agent"
if str(GEO_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(GEO_AGENT_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

from geo_agent.core.models import CitationCheckResult
from geo_agent.core.memory import parse_tool_output
from geo_agent.utils.structural_parser import ContentChunk, StructuralHtmlParser
from geo_agent.tools import registry

# 复用 geo_agent 核心模块（与 geo_agent 对齐）
from geo_agent.agent.content_auditor import audit_content_truncation, TruncationAuditResult
from geo_agent.agent.policy_engine import PolicyEngine as GeoAgentPolicyEngine
from geo_agent.core.telemetry import (
    TelemetryStore,
    FailureCategory,
    ToolInvocationSpan,
    IterationMetrics,
    ToolOutcome,
    compute_args_hash,
)

from .failure_analysis import analyze_failure_async, DiagnosisResult, regenerate_tool_args_async
from .memory_manager import OptimizationMemoryV2, PolicyEngine as BatchPolicyEngine, HistoryManagerV2
from .models import DiagnosisInfo, QueryResultV2, SuggestionV2
from .tool_executor import ToolExecutor
from .citation_checker import compute_geo_score, GEOScoreInfo

logger = logging.getLogger(__name__)


class SuggestionCollectorV2:
    """
    V2 建议收集器

    使用两阶段分析：
    1. 诊断（Diagnose）- 识别失败根因
    2. 策略选择（Select Tool Strategy）- 选择最佳工具

    支持：
    - 并行处理多个 query
    - 每个 query 多次重试
    - 诊断信息记录
    - 策略注入
    """

    def __init__(
        self,
        llm,
        generator: Any,  # AsyncInContextGenerator
        chunks: List[ContentChunk],
        core_ideas: Dict[int, str],
        history_context: str = "",
        chunks_per_orchestra: int = 2,
        history_manager: Optional[HistoryManagerV2] = None,
        enable_policy_injection: bool = True,
        enable_memory: bool = True,
        enable_history: bool = True,
        excluded_tools: Optional[List[str]] = None,
    ):
        self.llm = llm
        self.generator = generator
        self.chunks = chunks
        self.core_ideas = core_ideas
        self.history_context = history_context
        self.chunks_per_orchestra = chunks_per_orchestra
        self.tool_executor = ToolExecutor()
        self.history_manager = history_manager or HistoryManagerV2()
        self.enable_policy_injection = enable_policy_injection
        self.enable_memory = enable_memory
        self.enable_history = enable_history
        self.excluded_tools = excluded_tools or []

        # 每个 query 的内存（用于追踪重试）
        self._query_memories: Dict[str, OptimizationMemoryV2] = {}

    def _get_chunks_summary(self) -> str:
        """生成 chunks 摘要（用于 LLM 选择目标 chunk）"""
        summary_parts = []
        for i, chunk in enumerate(self.chunks):
            text_preview = chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
            summary_parts.append(f"[CHUNK {i}]: {text_preview}")
        return "\n\n".join(summary_parts)

    def _format_indexed_content(self) -> str:
        """生成带索引的完整内容"""
        return "\n\n".join(
            [f">> [CHUNK_ID: {i}]\n{chunk.text}" for i, chunk in enumerate(self.chunks)]
        )

    def _sanitize_for_logging(self, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """脱敏工具参数，移除过长内容（与 geo_agent 对齐）"""
        sanitized = {}
        for key, value in tool_args.items():
            if key in ["target_content", "context_before", "context_after"]:
                sanitized[key] = f"<{len(str(value))} chars>" if value else ""
            elif key == "previous_modifications":
                lines = str(value).strip().split('\n') if value else []
                sanitized[key] = f"<{len(lines)} rules>"
            else:
                sanitized[key] = value
        return sanitized

    def _str_to_failure_category(self, diagnosis_str: str) -> FailureCategory:
        """将字符串诊断转换为 FailureCategory 枚举"""
        try:
            return FailureCategory(diagnosis_str.lower())
        except ValueError:
            return FailureCategory.UNKNOWN

    def _get_orchestra_core_idea(self, chunk_index: int) -> str:
        """获取 chunk 所属 orchestra 的核心思想"""
        orchestra_id = chunk_index // self.chunks_per_orchestra
        return self.core_ideas.get(orchestra_id, "")

    async def collect_for_query(
        self,
        query: str,
        original_content: str,
        original_html: str,
        retrieved_docs: List[Any],
        competitor_contents: List[str],
        max_retries: int = 3,
        check_citation_func: Optional[Callable] = None,
    ) -> QueryResultV2:
        """
        为单个 query 收集建议（与 GEO Agent 完全一致的 retry 验证机制）

        V2.2 更新（与 geo_agent 对齐）：
        - 复用截断审计 audit_content_truncation()
        - 复用策略引擎 PolicyEngine 和遥测存储 TelemetryStore
        - 两阶段策略评估和工具覆盖
        - 快速失败检测

        核心逻辑：
        1. 创建临时 HTML 结构用于测试修改
        2. 每次迭代：导出HTML → 重新解析 → 获取纯文本 → 检查引用
        3. 失败则执行工具，更新临时结构
        4. 成功则退出，返回最终建议

        Args:
            query: 用户查询
            original_content: 原始文档内容（纯文本，不变）
            original_html: 原始 HTML（用于创建临时结构）
            retrieved_docs: 检索到的竞争文档
            competitor_contents: 竞争对手完整内容
            max_retries: 最大重试次数
            check_citation_func: 引用检查函数

        Returns:
            QueryResultV2: 查询结果（包含最终生效的建议和 final_html）
        """
        import time

        is_cited = False
        generated_answer = ""
        final_diagnosis: Optional[DiagnosisInfo] = None
        final_suggestion: Optional[SuggestionV2] = None
        iterations_used = 0
        # GEO Score 信息（V2.3 新增）
        final_geo_score: Optional[GEOScoreInfo] = None

        # 为该 query 创建内存
        memory = OptimizationMemoryV2()
        self._query_memories[query] = memory

        # ========== 初始化遥测存储和策略引擎（复用 geo_agent）==========
        telemetry = TelemetryStore(url="", core_idea=self.core_ideas.get(0, ""))
        geo_policy_engine = GeoAgentPolicyEngine(telemetry)

        # 保留原有的 BatchPolicyEngine 用于策略注入 prompt 生成
        batch_policy_engine = BatchPolicyEngine(
            self.history_manager,
            memory,
            enable_memory=self.enable_memory,
            enable_history=self.enable_history,
        )

        # ========== 双结构策略：frozen_structure（索引稳定）+ temp_structure（内容最新）==========
        struct_parser = StructuralHtmlParser(min_length=50)

        # frozen_structure: 只计算一次 chunks，用于稳定的索引映射
        frozen_structure = struct_parser.parse(original_html)
        frozen_structure.calculate_chunks(max_chunk_length=2000)
        frozen_num_chunks = len(frozen_structure._chunks)

        # JS Fallback: 当 chunks=0 时，创建虚拟 chunk 包含整个 HTML
        # 这样 static_rendering 工具可以尝试从 JS/JSON 中提取内容
        js_fallback_mode = False
        if frozen_num_chunks == 0 and original_html:
            from geo_agent.utils.structural_parser import ContentChunk
            # 注意：使用 'id' 键名（不是 'geo_id'）以匹配 apply_modification_to_live 的期望
            virtual_element = {
                'text_content': original_content if original_content else '',
                'original_html': original_html,
                'id': 'virtual-js-chunk-0',  # 用于标识虚拟 chunk
            }
            virtual_chunk = ContentChunk(index=0, elements=[virtual_element])
            frozen_structure._chunks = [virtual_chunk]
            frozen_num_chunks = 1
            js_fallback_mode = True
            logger.info(f"[JS Fallback] Created virtual chunk from raw HTML ({len(original_html)} chars)")

        # temp_structure: 动态更新的 DOM，用于内容最新状态
        temp_structure = struct_parser.parse(original_html)
        if js_fallback_mode:
            # 同步 temp_structure 的虚拟 chunk
            temp_structure._chunks = frozen_structure._chunks.copy()

        # 截断审计信息（跨迭代共享）
        truncation_summary: Optional[str] = None
        has_truncation_alert = False

        for iteration in range(max_retries):
            iterations_used = iteration + 1
            tool_start_time = time.time()
            tool_outcome = ToolOutcome.SKIPPED
            tool_error_msg: Optional[str] = None
            args_hash = ""

            logger.info(f"Query '{query[:50]}...' - Iteration {iteration}/{max_retries}")

            try:
                # ========== 1. 刷新结构（和 GEO Agent 完全一致）==========
                current_raw_html = temp_structure.export_html()
                temp_structure = struct_parser.parse(current_raw_html)  # 重新解析
                temp_content = temp_structure.get_clean_text()  # HTML → 纯文本

                # 如果解析后内容为空，使用原始内容并跳过引用检查
                content_empty = not temp_content or not temp_content.strip()
                if content_empty:
                    logger.warning(f"Empty content after parsing at iteration {iteration}, using original for analysis")
                    temp_content = original_content  # 用于后续截断审计
                    is_cited = False
                    cited_indices = []
                elif check_citation_func:
                    citation_result: CitationCheckResult = await check_citation_func(
                        query, temp_content, retrieved_docs, competitor_contents
                    )
                    is_cited = citation_result.is_cited
                    generated_answer = citation_result.generated_answer
                    cited_indices = citation_result.citations_found_idx

                    # 计算 GEO Score（V2.3 新增）
                    # target_idx 默认为 competitor_contents 数量 + 1（目标文档在最后）
                    num_sources = len(competitor_contents) + 1
                    target_idx = num_sources  # 假设目标文档是最后一个
                    final_geo_score = compute_geo_score(generated_answer, target_idx, num_sources)

                    if is_cited:
                        logger.info(f"✅ Query '{query[:50]}...' - Cited at iteration {iteration}!")
                        logger.info(f"📊 GEO Score: overall={final_geo_score.overall:.4f}, word={final_geo_score.word:.4f}, pos={final_geo_score.position:.4f}")
                        break
                else:
                    cited_indices = []

                # ========== 3. 准备分析所需内容 ==========
                # 使用双结构策略：索引来自 frozen_structure（稳定），内容来自 temp_structure（最新）
                indexed_content = frozen_structure.format_indexed_content_with_live_dom(temp_structure)
                num_chunks = frozen_num_chunks  # 始终使用冻结的 chunk 数量

                # ========== 3.1 截断审计（复用 geo_agent）==========
                if iteration == 0:  # 只在第一次迭代执行截断审计
                    try:
                        audit_res = audit_content_truncation(
                            self.llm,
                            query,
                            full_text=temp_content,
                            visible_chunks_text=indexed_content
                        )
                        if audit_res.has_hidden_relevant_content:
                            logger.info(f"⚠️ Truncation Alert: {audit_res.summary_of_hidden_info}")
                            truncation_summary = audit_res.summary_of_hidden_info
                            has_truncation_alert = True
                    except Exception as e:
                        logger.warning(f"Truncation audit failed: {e}")

                # 准备竞争对手内容
                # 注意: cited_indices 是 1-based (LLM 生成 [1], [2] 等)
                if cited_indices and competitor_contents:
                    valid_indices = [i for i in cited_indices if 1 <= i <= len(competitor_contents)]
                    if valid_indices:
                        competitor_content = "\n---\n".join(
                            [competitor_contents[i - 1][:3000] for i in valid_indices[:3]]  # i-1 转为 0-based
                        )
                    else:
                        competitor_content = competitor_contents[0][:3000] if competitor_contents else ""
                else:
                    competitor_content = competitor_contents[0][:3000] if competitor_contents else ""

                # ========== 4. Phase 1 策略评估（诊断前，基于截断信息）==========
                # JS Fallback 模式：强制使用 PARSING_FAILURE 诊断
                if js_fallback_mode:
                    pre_diagnosis_category = FailureCategory.PARSING_FAILURE
                    logger.info("[JS Fallback] Using PARSING_FAILURE diagnosis for JS/JSON content")
                elif has_truncation_alert:
                    pre_diagnosis_category = FailureCategory.CONTENT_TRUNCATED
                else:
                    pre_diagnosis_category = FailureCategory.UNKNOWN

                pre_policy_eval = geo_policy_engine.evaluate(
                    diagnosis_category=pre_diagnosis_category,
                    diagnosis_explanation="",
                    has_truncation_alert=has_truncation_alert,
                    hidden_content_summary=truncation_summary or ""
                )

                # ========== 4.1 生成策略注入（合并 geo_agent 策略和 batch 策略）==========
                policy_injection = ""
                if self.enable_policy_injection:
                    # 优先使用 geo_agent 策略引擎的注入
                    if pre_policy_eval.injection_prompt:
                        policy_injection = pre_policy_eval.injection_prompt
                    elif self.enable_memory:
                        # Fallback 到 batch_policy_engine（仅当 enable_memory=True 时）
                        policy_injection = batch_policy_engine.generate_policy_injection(
                            current_diagnosis=final_diagnosis,
                            current_chunk_index=None,
                        )

                # ========== 5. 两阶段分析 ==========
                analysis, diagnosis = await analyze_failure_async(
                    llm=self.llm,
                    query=query,
                    indexed_target_doc=indexed_content,
                    competitor_doc=competitor_content,
                    memory=memory,
                    truncation_audit_summary=truncation_summary,  # 传递截断信息
                    policy_injection=policy_injection,
                    num_chunks=num_chunks,
                    excluded_tools=self.excluded_tools,
                )

                # 记录诊断
                diagnosis_info = diagnosis.to_diagnosis_info()
                final_diagnosis = diagnosis_info
                logger.info(f"Diagnosis: {diagnosis.root_cause} - {diagnosis.key_deficiency}")

                # JS Fallback 模式：强制使用 static_rendering 工具（仅第一次迭代）
                if js_fallback_mode and iteration == 0 and "static_rendering" not in self.excluded_tools:
                    original_tool = analysis.selected_tool_name
                    analysis.selected_tool_name = "static_rendering"
                    analysis.tool_arguments = {}  # static_rendering 不需要额外参数
                    logger.info(f"[JS Fallback] Overriding tool: {original_tool} -> static_rendering")

                logger.info(f"Tool Selected: {analysis.selected_tool_name}")

                # ========== 5.1 Phase 2 策略评估（诊断后）==========
                diagnosis_category = self._str_to_failure_category(diagnosis.root_cause)

                policy_eval = geo_policy_engine.evaluate(
                    diagnosis_category=diagnosis_category,
                    diagnosis_explanation=diagnosis.explanation,
                    has_truncation_alert=has_truncation_alert,
                    hidden_content_summary=truncation_summary or "",
                    severity=diagnosis.severity
                )

                # ========== 5.2 应用强制工具覆盖（修复：重新生成参数）==========
                # 注意：JS Fallback 模式下的 static_rendering 工具不应被覆盖
                original_tool = analysis.selected_tool_name
                skip_policy_override = (js_fallback_mode and analysis.selected_tool_name == "static_rendering")
                if policy_eval.forced_tool and policy_eval.forced_tool != analysis.selected_tool_name and not skip_policy_override:
                    logger.info(f"🎯 Policy Override: {analysis.selected_tool_name} -> {policy_eval.forced_tool}")

                    # 重新生成适配新工具的参数（修复参数不匹配问题）
                    try:
                        history_context = memory.get_history_summary() if self.enable_memory and memory else ""
                        analysis = await regenerate_tool_args_async(
                            llm=self.llm,
                            forced_tool=policy_eval.forced_tool,
                            diagnosis=diagnosis,
                            query=query,
                            target_content_indexed=indexed_content,
                            history_context=history_context,
                            num_chunks=num_chunks,
                        )
                        logger.info(f"✅ Regenerated args for {policy_eval.forced_tool}")
                    except Exception as e:
                        logger.error(f"Failed to regenerate args for {policy_eval.forced_tool}: {e}")
                        # 回退到原始工具
                        analysis.selected_tool_name = original_tool
                        logger.warning(f"⚠️ Falling back to original tool: {original_tool}")

                # ========== 5.3 检查工具是否被禁止 ==========
                if analysis.selected_tool_name in policy_eval.blocked_tools:
                    logger.warning(f"Tool {analysis.selected_tool_name} is blocked by policy, trying next iteration")
                    tool_outcome = ToolOutcome.SKIPPED
                    continue

                # ========== 5.4 快速失败检测（复用 geo_agent 的判断逻辑）==========
                is_fixable, fixable_reason = geo_policy_engine.is_category_fixable(diagnosis_category)
                if not is_fixable and diagnosis.severity == "critical":
                    logger.warning(f"⚡ Unfixable diagnosis: {diagnosis.root_cause} - {fixable_reason}")
                    break

                if policy_eval.should_skip:
                    logger.warning(f"⚡ Policy suggests skip: {policy_eval.skip_reason}")
                    if iteration >= 1:  # 已尝试至少一次
                        break

                # ========== 6. 准备工具参数（双结构策略）==========
                target_chunk_index = analysis.target_chunk_index or 0
                if target_chunk_index >= num_chunks:
                    target_chunk_index = num_chunks - 1

                # 使用双结构策略：frozen_structure 的索引定位，temp_structure 的最新内容
                tool_args = analysis.tool_arguments.copy()
                tool_args.update(frozen_structure.get_chunk_tool_args_from_live(temp_structure, target_chunk_index))

                # 获取 orchestra 的核心思想
                orchestra_id = target_chunk_index // self.chunks_per_orchestra
                core_idea = self.core_ideas.get(orchestra_id, "")
                tool_args['core_idea'] = core_idea
                tool_args['previous_modifications'] = memory.get_preservation_rules() if self.enable_memory else ""

                # 为 content_relocation 工具添加必需参数 (和 geo_agent 一致)
                if analysis.selected_tool_name == "content_relocation":
                    # 仅当 truncation_summary 有实际内容时才覆盖，否则保留 LLM 可能生成的值
                    if truncation_summary:
                        tool_args["hidden_content_summary"] = truncation_summary
                    elif "hidden_content_summary" not in tool_args:
                        tool_args["hidden_content_summary"] = ""
                    tool_args["query"] = query

                # 为 intent_realignment 工具添加必需的 user_query 参数
                if analysis.selected_tool_name == "intent_realignment":
                    tool_args["user_query"] = query

                # 为 historical_redteam 工具添加必需的 target_query 参数
                if analysis.selected_tool_name == "historical_redteam":
                    tool_args["target_query"] = query

                # ========== 6.1 检查重复调用（与 geo_agent 完全一致）==========
                args_hash = compute_args_hash(tool_args)
                is_dup, dup_msg = geo_policy_engine.check_duplicate_invocation(analysis.selected_tool_name, args_hash)
                if is_dup:
                    logger.warning(dup_msg)
                    tool_outcome = ToolOutcome.SKIPPED
                    continue

                # ========== 7. 执行工具 ==========
                tool = registry.get_tool(analysis.selected_tool_name)
                if not tool:
                    logger.error(f"Tool {analysis.selected_tool_name} not found.")
                    tool_outcome = ToolOutcome.FAILED
                    tool_error_msg = f"Tool {analysis.selected_tool_name} not found"
                    continue

                try:
                    raw_output = await asyncio.to_thread(tool.run, tool_args)
                    modified_chunk_html, key_changes = parse_tool_output(raw_output)
                    tool_outcome = ToolOutcome.SUCCESS

                    # ========== 8. 更新临时结构（双结构策略：通过 frozen 索引定位，在 temp 上修改）==========
                    if js_fallback_mode:
                        # JS Fallback 模式：虚拟 chunk 没有真正的 DOM 锚点
                        # 直接用工具输出的新 HTML 替换整个结构
                        wrapped_html = f"<html><body>{modified_chunk_html}</body></html>"
                        temp_structure = struct_parser.parse(wrapped_html)
                        temp_structure.calculate_chunks(max_chunk_length=2000)
                        logger.info(f"[JS Fallback] Replaced temp_structure with tool output ({len(modified_chunk_html)} chars)")
                    elif frozen_structure.apply_modification_to_live(temp_structure, target_chunk_index, modified_chunk_html, highlight=False):
                        logger.info(f"DOM updated successfully at frozen chunk index {target_chunk_index}")
                    else:
                        logger.warning(f"Failed to update DOM at frozen chunk index {target_chunk_index}")
                        tool_outcome = ToolOutcome.PARTIAL

                    # 创建建议记录（包含新增字段）
                    final_suggestion = SuggestionV2(
                        suggestion_id=str(uuid.uuid4())[:8],
                        query=query,
                        tool_name=analysis.selected_tool_name,
                        tool_arguments=analysis.tool_arguments,  # LLM 原始输出
                        target_segment_index=target_chunk_index,
                        reasoning=analysis.reasoning,
                        proposed_content=modified_chunk_html,
                        key_changes=key_changes,
                        diagnosis=diagnosis_info,
                        iteration=iteration,
                        confidence=self._calculate_confidence(diagnosis_info),
                        executed_arguments=self._sanitize_for_logging(tool_args),  # 实际执行参数（脱敏）
                        truncation_info={
                            "has_alert": has_truncation_alert,
                            "summary": truncation_summary
                        } if has_truncation_alert else None,
                    )

                    # 更新内存（仅当 enable_memory=True 时）
                    if self.enable_memory:
                        from .memory_manager import ModificationRecordV2
                        record = ModificationRecordV2(
                            query=query,
                            tool_name=analysis.selected_tool_name,
                            reasoning=analysis.reasoning,
                            key_changes=key_changes,
                            diagnosis=diagnosis_info,
                            chunk_index=target_chunk_index,
                        )
                        memory.add_modification(record)

                    logger.info(f"Tool '{analysis.selected_tool_name}' executed, changes: {key_changes}")

                except Exception as e:
                    logger.error(f"Tool execution failed: {e}")
                    import traceback
                    traceback.print_exc()
                    tool_outcome = ToolOutcome.FAILED
                    tool_error_msg = str(e)

                # ========== 9. 记录遥测数据（与 geo_agent 完全一致）==========
                tool_duration = (time.time() - tool_start_time) * 1000
                tool_span = ToolInvocationSpan(
                    tool_name=analysis.selected_tool_name,
                    target_chunk_index=target_chunk_index,
                    arguments_hash=args_hash,
                    outcome=tool_outcome,
                    reasoning=analysis.reasoning,
                    duration_ms=tool_duration,
                    error_message=tool_error_msg
                )

                iteration_metrics = IterationMetrics(
                    iteration_index=iteration,
                    query=query,
                    full_doc_length=len(temp_content),
                    visible_chunk_length=len(indexed_content),
                    truncation_ratio=1 - len(indexed_content) / max(len(temp_content), 1),
                    chunk_count=num_chunks,
                    diagnosis_category=diagnosis_category,
                    diagnosis_explanation=diagnosis.explanation,
                    has_hidden_relevant_content=has_truncation_alert,
                    hidden_content_summary=truncation_summary or "",
                    tool_span=tool_span,
                    was_cited=is_cited
                )
                telemetry.record_iteration(iteration_metrics)

            except Exception as e:
                logger.error(f"Analysis failed for query '{query[:50]}...': {e}")
                import traceback
                traceback.print_exc()
                continue

        # 获取最终的 HTML
        final_html = temp_structure.export_html() if final_suggestion else None

        return QueryResultV2(
            query=query,
            is_cited=is_cited,
            generated_answer=generated_answer,
            suggestions=[final_suggestion] if final_suggestion else [],
            diagnosis=final_diagnosis,
            iterations_used=iterations_used,
            final_html=final_html,
            # GEO Score 字段（V2.3 新增）
            geo_score_word=final_geo_score.word if final_geo_score else 0.0,
            geo_score_position=final_geo_score.position if final_geo_score else 0.0,
            geo_score_wordpos=final_geo_score.wordpos if final_geo_score else 0.0,
            geo_score_overall=final_geo_score.overall if final_geo_score else 0.0,
            has_valid_citations=final_geo_score.has_valid_citations if final_geo_score else False,
        )

    def _calculate_confidence(self, diagnosis: DiagnosisInfo) -> float:
        """根据诊断计算置信度"""
        severity_scores = {
            "critical": 0.9,
            "high": 0.8,
            "medium": 0.7,
            "low": 0.6,
        }
        return severity_scores.get(diagnosis.severity, 0.7)

    async def collect_batch(
        self,
        queries: List[str],
        current_content: str,
        current_html: str,
        retrieved_docs_func: Callable,
        competitor_contents_func: Callable,
        check_citation_func: Callable,
        max_concurrency: int = 4,
        max_retries_per_query: int = 3,
    ) -> List[QueryResultV2]:
        """
        批量收集建议

        V2.1 更新：
        - 传递 original_html 给每个 query 的处理
        - 每个 query 独立在临时结构上测试修改

        Args:
            queries: 查询列表
            current_content: 当前内容（纯文本）
            current_html: 当前 HTML（必需，用于创建临时结构）
            retrieved_docs_func: 获取检索文档的函数 (query) -> List[SearchResult]
            competitor_contents_func: 获取竞争对手内容的函数 (docs) -> List[str]
            check_citation_func: 检查引用的函数
            max_concurrency: 最大并发数
            max_retries_per_query: 每个 query 的最大重试次数

        Returns:
            List[QueryResultV2]: 查询结果列表（每个包含 final_html）
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def process_query(query: str) -> QueryResultV2:
            async with semaphore:
                try:
                    # 获取检索文档
                    retrieved_docs = await retrieved_docs_func(query)
                    if not retrieved_docs:
                        return QueryResultV2(
                            query=query,
                            is_cited=False,
                            generated_answer="",
                            suggestions=[],
                            error="No retrieved documents",
                        )

                    # 获取竞争对手内容（返回过滤后的 docs 和 contents）
                    result = await competitor_contents_func(retrieved_docs)
                    if isinstance(result, tuple) and len(result) == 2:
                        retrieved_docs, competitor_contents = result
                    else:
                        competitor_contents = result
                    if not competitor_contents:
                        return QueryResultV2(
                            query=query,
                            is_cited=False,
                            generated_answer="",
                            suggestions=[],
                            error="No competitor contents",
                        )

                    # 收集建议（传递 HTML）
                    return await self.collect_for_query(
                        query=query,
                        original_content=current_content,
                        original_html=current_html,
                        retrieved_docs=retrieved_docs,
                        competitor_contents=competitor_contents,
                        max_retries=max_retries_per_query,
                        check_citation_func=check_citation_func,
                    )
                except Exception as e:
                    logger.error(f"Failed to process query '{query[:50]}...': {e}")
                    import traceback
                    traceback.print_exc()
                    return QueryResultV2(
                        query=query,
                        is_cited=False,
                        generated_answer="",
                        suggestions=[],
                        error=str(e),
                    )

        tasks = [asyncio.create_task(process_query(q)) for q in queries]
        results = await asyncio.gather(*tasks)

        return results

    def get_all_suggestions(self) -> List[SuggestionV2]:
        """获取所有收集到的建议"""
        all_suggestions = []
        for memory in self._query_memories.values():
            # 从内存中提取建议
            pass
        return all_suggestions

    def get_diagnosis_stats(self) -> Dict[str, int]:
        """获取诊断统计"""
        stats: Dict[str, int] = {}
        for memory in self._query_memories.values():
            for cause, count in memory.diagnosis_stats.items():
                stats[cause] = stats.get(cause, 0) + count
        return stats
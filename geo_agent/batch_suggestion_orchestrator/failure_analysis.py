"""
Batch GEO V2 失败分析模块
异步版本的两阶段分析：诊断（Diagnose）+ 策略选择（Select Tool Strategy）

完全基于 geo_agent/agent/failure_analysis.py 实现
"""
import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# 设置路径
REPO_ROOT = Path(__file__).resolve().parents[1]
GEO_AGENT_ROOT = REPO_ROOT / "geo_agent"
if str(GEO_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(GEO_AGENT_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

from geo_agent.core.models import AnalysisResult
from geo_agent.core.memory import OptimizationMemory
from geo_agent.tools import registry

from .models import DiagnosisInfo

logger = logging.getLogger(__name__)


# 定义允许的失败类型字面量（与 geo_agent 保持一致）
FAILURE_CATEGORY_LITERAL = [
    "PARSING_FAILURE",
    "CONTENT_TRUNCATED",
    "DATA_INTEGRITY",
    "WEB_NOISE",
    "LOW_SIGNAL_RATIO",
    "LOW_INFO_DENSITY",
    "MISSING_INFO",
    "STRUCTURAL_WEAKNESS",
    "SEMANTIC_IRRELEVANCE",
    "ATTRIBUTE_MISMATCH",
    "BURIED_ANSWER",
    "NON_FACTUAL_CONTENT",
    "TRUST_CREDIBILITY",
    "OUTDATED_CONTENT",
    "UNKNOWN",
]


class DiagnosisResult(BaseModel):
    """诊断结果模型"""

    root_cause: str = Field(
        description="The failure category. MUST be one of the predefined values: "
        "PARSING_FAILURE, CONTENT_TRUNCATED, DATA_INTEGRITY, WEB_NOISE, "
        "LOW_SIGNAL_RATIO, LOW_INFO_DENSITY, MISSING_INFO, STRUCTURAL_WEAKNESS, "
        "SEMANTIC_IRRELEVANCE, ATTRIBUTE_MISMATCH, BURIED_ANSWER, NON_FACTUAL_CONTENT, "
        "TRUST_CREDIBILITY, OUTDATED_CONTENT, UNKNOWN"
    )
    explanation: str = Field(description="Detailed explanation of the failure.")
    key_deficiency: str = Field(description="What specific element is missing or weak.")
    severity: str = Field(default="medium", description="Severity level: low, medium, high, critical")

    def to_diagnosis_info(self) -> DiagnosisInfo:
        """转换为 DiagnosisInfo 数据类"""
        return DiagnosisInfo(
            root_cause=self.root_cause,
            explanation=self.explanation,
            key_deficiency=self.key_deficiency,
            severity=self.severity,
        )


# 完整的失败原因分类表（用于 Prompt）
FAILURE_TAXONOMY = """
### FAILURE CATEGORY TAXONOMY (Choose ONE that best describes the root cause):

**TECHNICAL ISSUES**
1. `PARSING_FAILURE` - Document is technically unreadable, corrupted, or poorly parsed due to crawl, parsing, rendering, or formatting failures
2. `CONTENT_TRUNCATED` - Document is incomplete, truncated, partial, or corrupted, missing substantial portions of relevant content
3. `DATA_INTEGRITY` - Document suffers from data integrity issues or crawling/extraction failures that reduce completeness, utility, or factual accuracy

**NOISE ISSUES**
4. `WEB_NOISE` - Document content consists primarily of non-informational web noise, boilerplate, navigational elements, or generic webpage components
5. `LOW_SIGNAL_RATIO` - Document is rejected due to semantic shortcomings, poor content quality, or low signal-to-noise ratio

**INFORMATION DENSITY ISSUES**
6. `LOW_INFO_DENSITY` - Document lacks sufficient information density, completeness, or granularity, providing minimal, sparse, anecdotal, or non-actionable content
7. `MISSING_INFO` - Document lacks specific facts, entities, or data points that the competitor has

**STRUCTURE ISSUES**
8. `STRUCTURAL_WEAKNESS` - Document is poorly structured, lacking clear segmentation, logical flow, thematic organization, or structural elements

**RELEVANCE ISSUES**
9. `SEMANTIC_IRRELEVANCE` - Document is semantically irrelevant, off-topic, topically unrelated, or misaligned with the user query
10. `ATTRIBUTE_MISMATCH` - Document addresses correct entity/topic but focuses on irrelevant, incorrect, or unrelated attributes, metrics, or aspects

**ANSWER POSITIONING ISSUES**
11. `BURIED_ANSWER` - Document fails to provide direct, explicit, or focused answers, burying relevant information deep within verbose, redundant content

**CONTENT QUALITY ISSUES**
12. `NON_FACTUAL_CONTENT` - Document contains only questions, speculations, opinions, filler, marketing, nostalgic, or anecdotal content without factual information
13. `TRUST_CREDIBILITY` - Document lacks authority, credibility, or trustworthy sources

**TEMPORAL ISSUES**
14. `OUTDATED_CONTENT` - Document is outdated, temporally mismatched, or lacks timely data required to satisfy the query context
"""


async def _ainvoke_llm(llm, prompt_input: str):
    """异步调用 LLM"""
    if hasattr(llm, "ainvoke"):
        return await llm.ainvoke(prompt_input)
    return await asyncio.to_thread(llm.invoke, prompt_input)


async def diagnose_root_cause_async(
    llm,
    query: str,
    target_content_preview: str,
    competitor_content: str,
    truncation_info: Optional[str] = None,
) -> DiagnosisResult:
    """
    Phase 1: 异步诊断（Diagnosis）
    识别文档未被引用的根本原因

    Args:
        llm: LLM 实例
        query: 用户查询
        target_content_preview: 目标文档预览
        competitor_content: 竞争对手文档内容
        truncation_info: 截断信息（可选）

    Returns:
        DiagnosisResult: 诊断结果
    """
    # Fast path: 如果检测到截断，直接返回
    if truncation_info:
        return DiagnosisResult(
            root_cause="CONTENT_TRUNCATED",
            explanation=f"Critical information was found but it is located too deep in the document (truncated view). {truncation_info}",
            key_deficiency="Visibility of existing content",
            severity="high",
        )

    parser = PydanticOutputParser(pydantic_object=DiagnosisResult)

    prompt = ChatPromptTemplate.from_template(
        """
    You are a Search Failure Analyst with expertise in document quality assessment.
    Your task is to determine the ROOT CAUSE of why the Target document was NOT cited while the Competitor was.

    Query: "{query}"

    === COMPETITOR DOCUMENT (Winner - Was Cited) ===
    {competitor}
    === END COMPETITOR ===

    === TARGET DOCUMENT (Loser - Not Cited) ===
    {target}
    === END TARGET ===

    {failure_taxonomy}

    ### ANALYSIS INSTRUCTIONS:
    1. **Compare** the two documents carefully against the query
    2. **Identify** the PRIMARY reason the Target failed
    3. **root_cause MUST be EXACTLY one of**: PARSING_FAILURE, CONTENT_TRUNCATED, DATA_INTEGRITY, WEB_NOISE, LOW_SIGNAL_RATIO, LOW_INFO_DENSITY, MISSING_INFO, STRUCTURAL_WEAKNESS, SEMANTIC_IRRELEVANCE, ATTRIBUTE_MISMATCH, BURIED_ANSWER, NON_FACTUAL_CONTENT, TRUST_CREDIBILITY, OUTDATED_CONTENT, UNKNOWN
    4. **Be Specific** in explanation - identify WHAT specifically is missing or wrong

    ### SEVERITY GUIDE:
    - `critical`: Document is fundamentally unusable (parsing failure, completely irrelevant)
    - `high`: Major issue requiring significant changes (truncation, buried answers, missing key info)
    - `medium`: Moderate issue requiring targeted fixes (structure, density)
    - `low`: Minor issue requiring polish (formatting, minor noise)

    ### CRITICAL: Output root_cause as EXACT category name (e.g., "MISSING_INFO", not "Missing Information")

    {format_instructions}
    """
    )

    final_prompt = prompt.format(
        query=query,
        competitor=competitor_content[:6000],
        target=target_content_preview[:6000],
        failure_taxonomy=FAILURE_TAXONOMY,
        format_instructions=parser.get_format_instructions(),
    )

    try:
        res = await _ainvoke_llm(llm, final_prompt)
        return parser.parse(res.content)
    except Exception as e:
        logger.warning(f"Diagnosis parsing failed: {e}")
        return DiagnosisResult(
            root_cause="UNKNOWN",
            explanation="Complex failure - could not parse diagnosis",
            key_deficiency="General quality",
            severity="medium",
        )


async def select_tool_strategy_async(
    llm,
    query: str,
    diagnosis: DiagnosisResult,
    target_content_indexed: str,
    available_tools_desc: str,
    history_context: str,
    memory: Optional[OptimizationMemory],
    policy_injection: str = "",
    num_chunks: Optional[int] = None,
    enable_autogeo_policy: bool = True,
) -> AnalysisResult:
    """
    Phase 2: 异步策略选择（Tool Selection）
    基于诊断结果选择最佳工具

    Args:
        llm: LLM 实例
        query: 用户查询
        diagnosis: 诊断结果
        target_content_indexed: 带索引的目标文档内容
        available_tools_desc: 可用工具描述
        history_context: 历史上下文
        memory: 优化记忆
        policy_injection: 策略注入
        num_chunks: chunk 数量（用于 Batch 模式）

    Returns:
        AnalysisResult: 分析结果
    """
    # 如果有 PolicyEngine 注入的规则，优先使用
    if policy_injection:
        policy_guidelines = policy_injection
    else:
        # Fallback: 默认的 Prompt 规则
        policy_guidelines = """
        ### OPTIMIZATION POLICY (Follow Strictly):
        0. IF Diagnosis is 'PARSING_FAILURE' (JS/JSON content) -> YOU MUST use 'static_rendering' to extract readable content from embedded data.
        1. IF Diagnosis is 'CONTENT_TRUNCATED' -> YOU MUST use 'content_relocation' to surface hidden content.
        2. IF Diagnosis is 'MISSING_INFO' AND History shows 'entity_injection' was already tried -> Switch strategy to 'persuasive_rewriting' or 'historical_redteam'.
        3. IF Diagnosis is 'STRUCTURAL_WEAKNESS' -> Prioritize 'structure_optimization' or 'intent_realignment'.
        4. IF Diagnosis is 'BURIED_ANSWER' -> Use 'content_relocation' to move relevant content to the top.
        5. IF Diagnosis is 'LOW_INFO_DENSITY' -> Use 'entity_injection' to add specific facts.
        6. NEVER repeat the exact same tool and arguments if it failed previously.
        """
        # 只有启用 autogeo_rephrase 时才添加相关策略
        if enable_autogeo_policy:
            policy_guidelines += """
        ### AUTOGEO META-TOOL POLICY:
        7. IF Diagnosis shows MULTIPLE issues (e.g., LOW_INFO_DENSITY + STRUCTURAL_WEAKNESS + TRUST_CREDIBILITY) -> Consider 'autogeo_rephrase' for comprehensive rewriting.
        8. IF History shows 2+ different tools have been tried without success -> Use 'autogeo_rephrase' as the "ultimate tool" for holistic optimization.
        9. IF Diagnosis is 'TRUST_CREDIBILITY' with severity 'high' or 'critical' -> Consider 'autogeo_rephrase' for academic-validated credibility enhancement.
        """

    analysis_parser = PydanticOutputParser(pydantic_object=AnalysisResult)

    # 根据是否有多个 chunks 调整 prompt
    chunk_instruction = ""
    if num_chunks and num_chunks > 1:
        chunk_instruction = f"""
    ### CHUNK SELECTION:
    The document has been divided into {num_chunks} chunks (0-indexed).
    You MUST include `"target_chunk_index": <int>` in your output to specify which chunk to modify.
    """

    prompt = ChatPromptTemplate.from_template(
        """
    You are the GEO Optimization Controller. Select the best tool based on the Diagnosis and Policy.

    ### Inputs
    - **Query**: {query}
    - **Root Cause Diagnosis**: {diagnosis_cause} ({diagnosis_explanation})
    - **Key Deficiency**: {key_deficiency}
    - **Severity**: {severity}
    - **History**: {history_context}

    ### Policy
    {policy_guidelines}

    {chunk_instruction}

    ### Target Document (Indexed)
    {target_content}

    ### Available Tools
    {tool_descriptions}

    ### Task
    Select the tool to resolve the '{diagnosis_cause}'.
    Identify the specific chunk/section that needs modification.

    ### Constraints
    - **Targeting:** You must include `"target_chunk_index": <int>` in your output.
    - **Injection Rule:** Include ["target_content", "context_before", "context_after", "core_idea", "previous_modifications"] in `tool_arguments` with empty string values (""). The system will populate these.

    {format_instructions}
    """
    )

    final_prompt = prompt.format(
        query=query,
        diagnosis_cause=diagnosis.root_cause,
        diagnosis_explanation=diagnosis.explanation,
        key_deficiency=diagnosis.key_deficiency,
        severity=diagnosis.severity,
        history_context=history_context,
        policy_guidelines=policy_guidelines,
        chunk_instruction=chunk_instruction,
        target_content=target_content_indexed,
        tool_descriptions=available_tools_desc,
        format_instructions=analysis_parser.get_format_instructions(),
    )

    # 重试逻辑
    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        try:
            res = await _ainvoke_llm(llm, final_prompt)
            return analysis_parser.parse(res.content)
        except Exception as e:
            last_error = e
            logger.warning(f"Tool selection parse attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                final_prompt += (
                    "\n\nIMPORTANT: Your response MUST include all required fields: "
                    "reasoning, selected_tool_name, AND tool_arguments (as a JSON object with the tool's required parameters)."
                )

    # 如果所有重试都失败，抛出异常
    raise last_error


async def analyze_failure_async(
    llm,
    query: str,
    indexed_target_doc: str,
    competitor_doc: str,
    memory: Optional[OptimizationMemory] = None,
    truncation_audit_summary: Optional[str] = None,
    policy_injection: str = "",
    num_chunks: Optional[int] = None,
    excluded_tools: Optional[List[str]] = None,
) -> Tuple[AnalysisResult, DiagnosisResult]:
    """
    完整的两阶段异步失败分析

    1. 诊断（Diagnose）- 识别失败根因
    2. 策略选择（Select Tool Strategy）- 选择最佳工具

    Args:
        llm: LLM 实例
        query: 用户查询
        indexed_target_doc: 带索引的目标文档
        competitor_doc: 竞争对手文档
        memory: 优化记忆（可选）
        truncation_audit_summary: 截断审计摘要（可选）
        policy_injection: 策略注入（可选）
        num_chunks: chunk 数量（用于 Batch 模式）
        excluded_tools: 要排除的工具名列表（如 ["autogeo_rephrase"]）

    Returns:
        Tuple[AnalysisResult, DiagnosisResult]: (分析结果, 诊断结果)
    """
    # 1. 诊断
    diagnosis = await diagnose_root_cause_async(
        llm, query, indexed_target_doc, competitor_doc, truncation_audit_summary
    )
    logger.info(f"Diagnosis: {diagnosis.root_cause} - {diagnosis.key_deficiency}")

    # 2. 准备上下文
    history_context = ""
    if memory and memory.modifications:
        history_context = memory.get_history_summary()

    # 工具过滤
    tool_descs = get_tool_descriptions(excluded_tools)

    # 判断是否启用 autogeo_rephrase 策略
    enable_autogeo = excluded_tools is None or "autogeo_rephrase" not in excluded_tools

    # 3. 选择工具（带策略注入）
    analysis = await select_tool_strategy_async(
        llm,
        query,
        diagnosis,
        indexed_target_doc,
        tool_descs,
        history_context,
        memory,
        policy_injection=policy_injection,
        num_chunks=num_chunks,
        enable_autogeo_policy=enable_autogeo,
    )

    return analysis, diagnosis


def get_tool_descriptions(excluded_tools: Optional[List[str]] = None) -> str:
    """获取所有工具的描述

    Args:
        excluded_tools: 要排除的工具名列表（如 ["autogeo_rephrase"]）
    """
    excluded = set(excluded_tools or [])
    tools = [t for t in registry.get_all_tools() if t.name not in excluded]
    return "\n".join(
        [f"{t.name}: {t.description} - Args: {t.args_schema.schema_json()}" for t in tools]
    )


async def regenerate_tool_args_async(
    llm,
    forced_tool: str,
    diagnosis: DiagnosisResult,
    query: str,
    target_content_indexed: str,
    history_context: str = "",
    num_chunks: Optional[int] = None,
) -> AnalysisResult:
    """
    Policy Override 后重新生成工具参数

    当 Policy Engine 强制切换工具时，原始参数可能不匹配新工具的 schema。
    此函数重新调用 LLM，专门为指定工具生成正确格式的参数。

    Args:
        llm: LLM 实例
        forced_tool: Policy 强制指定的工具名
        diagnosis: 诊断结果
        query: 用户查询
        target_content_indexed: 带索引的目标文档
        history_context: 历史上下文
        num_chunks: chunk 数量

    Returns:
        AnalysisResult: 包含正确参数格式的分析结果
    """
    # 获取指定工具的 schema
    tool = registry.get_tool(forced_tool)
    if not tool:
        raise ValueError(f"Tool {forced_tool} not found in registry")

    tool_schema = tool.args_schema.schema_json()
    tool_desc = f"{tool.name}: {tool.description}"

    analysis_parser = PydanticOutputParser(pydantic_object=AnalysisResult)

    chunk_instruction = ""
    if num_chunks and num_chunks > 1:
        chunk_instruction = f"""
    ### CHUNK SELECTION:
    The document has been divided into {num_chunks} chunks (0-indexed).
    You MUST include `"target_chunk_index": <int>` in your output to specify which chunk to modify.
    """

    prompt = ChatPromptTemplate.from_template(
        """
    You are the GEO Optimization Controller. Generate the correct arguments for the SPECIFIED tool.

    ### IMPORTANT: Tool is ALREADY SELECTED
    You MUST use this tool: **{forced_tool}**
    DO NOT select a different tool. Only generate the correct arguments.

    ### Inputs
    - **Query**: {query}
    - **Root Cause Diagnosis**: {diagnosis_cause} ({diagnosis_explanation})
    - **Key Deficiency**: {key_deficiency}
    - **History**: {history_context}

    {chunk_instruction}

    ### Target Document (Indexed)
    {target_content}

    ### Tool to Use (MANDATORY)
    {tool_desc}

    ### Tool Arguments Schema (MUST FOLLOW EXACTLY)
    {tool_schema}

    ### Task
    1. Analyze the diagnosis and target content
    2. Generate the correct `tool_arguments` that match the schema above
    3. `selected_tool_name` MUST be "{forced_tool}"

    ### Constraints
    - **selected_tool_name**: MUST be exactly "{forced_tool}"
    - **tool_arguments**: MUST match the schema for {forced_tool}
    - Include ["target_content", "context_before", "context_after", "core_idea", "previous_modifications"] with empty string values (""). The system will populate these.

    {format_instructions}
    """
    )

    final_prompt = prompt.format(
        forced_tool=forced_tool,
        query=query,
        diagnosis_cause=diagnosis.root_cause,
        diagnosis_explanation=diagnosis.explanation,
        key_deficiency=diagnosis.key_deficiency,
        history_context=history_context,
        chunk_instruction=chunk_instruction,
        target_content=target_content_indexed[:8000],  # 限制长度
        tool_desc=tool_desc,
        tool_schema=tool_schema,
        format_instructions=analysis_parser.get_format_instructions(),
    )

    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        try:
            res = await _ainvoke_llm(llm, final_prompt)
            result = analysis_parser.parse(res.content)

            # 验证工具名是否正确
            if result.selected_tool_name != forced_tool:
                logger.warning(f"LLM returned wrong tool {result.selected_tool_name}, forcing to {forced_tool}")
                result.selected_tool_name = forced_tool

            return result
        except Exception as e:
            last_error = e
            logger.warning(f"Regenerate args attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                final_prompt += (
                    f"\n\nCRITICAL: Your response MUST have selected_tool_name=\"{forced_tool}\" "
                    f"and tool_arguments matching the {forced_tool} schema."
                )

    raise last_error

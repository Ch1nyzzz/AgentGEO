from typing import Optional, Dict, Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from geo_agent.core.models import AnalysisResult
from geo_agent.core.memory import OptimizationMemory
from geo_agent.core.telemetry import FailureCategory
from geo_agent.tools import registry


# Define allowed failure type literals
FAILURE_CATEGORY_LITERAL = Literal[
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
    "UNKNOWN"
]


class DiagnosisResult(BaseModel):
    root_cause: FAILURE_CATEGORY_LITERAL = Field(
        description="The failure category. MUST be one of the predefined values: PARSING_FAILURE, CONTENT_TRUNCATED, DATA_INTEGRITY, WEB_NOISE, LOW_SIGNAL_RATIO, LOW_INFO_DENSITY, MISSING_INFO, STRUCTURAL_WEAKNESS, SEMANTIC_IRRELEVANCE, ATTRIBUTE_MISMATCH, BURIED_ANSWER, NON_FACTUAL_CONTENT, TRUST_CREDIBILITY, OUTDATED_CONTENT, UNKNOWN"
    )
    explanation: str = Field(description="Detailed explanation of the failure.")
    key_deficiency: str = Field(description="What specific element is missing or weak.")
    severity: Literal["low", "medium", "high", "critical"] = Field(
        default="medium", 
        description="Severity level: low, medium, high, critical"
    )
    
    def to_category(self) -> FailureCategory:
        """Direct mapping from string to enum, no keyword matching needed"""
        try:
            return FailureCategory(self.root_cause.lower())
        except ValueError:
            return FailureCategory.UNKNOWN


# Complete failure reason taxonomy (for Prompt)
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

def diagnose_root_cause(
    llm,
    query: str,
    target_content_preview: str,
    competitor_content: str,
    truncation_info: Optional[str] = None
) -> DiagnosisResult:
    """
    Phase 1: Diagnosis (The 'Trace' & 'Metric').
    Identifies WHY we failed before deciding WHAT to do.
    Uses comprehensive failure taxonomy based on large-scale document analysis.
    """
    
    # Fast path: If truncation found relevant info, force that diagnosis.
    if truncation_info:
        return DiagnosisResult(
            root_cause="CONTENT_TRUNCATED",
            explanation=f"Critical information was found but it is located too deep in the document (truncated view). {truncation_info}",
            key_deficiency="Visibility of existing content",
            severity="high"
        )
        
    parser = PydanticOutputParser(pydantic_object=DiagnosisResult)
    
    prompt = ChatPromptTemplate.from_template("""
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
    """)
    
    final_prompt = prompt.format(
        query=query,
        competitor=competitor_content[:6000],  # Increased for better comparison
        target=target_content_preview[:6000],
        failure_taxonomy=FAILURE_TAXONOMY,
        format_instructions=parser.get_format_instructions()
    )
    
    try:
        res = llm.invoke(final_prompt)
        return parser.parse(res.content)
    except Exception as e:
        print(f"Diagnosis parsing failed: {e}")
        # Fallback
        return DiagnosisResult(
            root_cause="UNKNOWN", 
            explanation="Complex failure - could not parse diagnosis", 
            key_deficiency="General quality",
            severity="medium"
        )

def select_tool_strategy(
    llm,
    query: str,
    diagnosis: DiagnosisResult,
    target_content_indexed: str,
    available_tools_desc: str,
    history_context: str,
    memory: Optional[OptimizationMemory],
    policy_injection: str = ""
) -> AnalysisResult:
    """
    Phase 2: Policy Enforcement & Tool Selection.
    Uses 'Framework Injection' to guide the agent based on Diagnosis + History.
    
    Args:
        policy_injection: Mandatory rules injected from PolicyEngine
    """

    # If there are rules injected from PolicyEngine, use them first
    if policy_injection:
        policy_guidelines = policy_injection
    else:
        # Fallback: Default prompt rules
        policy_guidelines = """
        ### OPTIMIZATION POLICY (Follow Strictly):
        1. IF Diagnosis is 'Content Truncated' -> YOU MUST use 'content_relocation' to surface hidden content.
        2. IF Diagnosis is 'Missing Information' AND History shows 'entity_injection' was already tried -> Switch strategy to 'persuasive_rewriting' or 'historical_redteam'.
        3. IF Diagnosis is 'Structural Issue' -> Prioritize 'structure_optimization' or 'intent_realignment'.
        4. NEVER repeat the exact same tool and arguments if it failed previously.
        """
    
    analysis_parser = PydanticOutputParser(pydantic_object=AnalysisResult)
    
    prompt = ChatPromptTemplate.from_template("""
    You are the GEO Optimization Controller. Select the best tool based on the Diagnosis and Policy.
    
    ### Inputs
    - **Query**: {query}
    - **Root Cause Diagnosis**: {diagnosis_cause} ({diagnosis_explanation})
    - **History**: {history_context}
    
    ### Policy
    {policy_guidelines}
    
    ### Target Document (Indexed)
    {target_content}
    
    ### Available Tools
    {tool_descriptions}
    
    ### Task
    Select the tool to resolve the '{diagnosis_cause}'.
    Identify the specific [CHUNK_ID] that needs modification.
    
    ### Constraints
    - **Targeting:** You must include `"target_chunk_index": <int>` in your output.
    - **Injection Rule:** Include ["target_content", "context_before", "context_after", "core_idea", "previous_modifications"] in `tool_arguments` with empty string values (""). The system will populate these.
    
    {format_instructions}
    """)
    
    final_prompt = prompt.format(
        query=query,
        diagnosis_cause=diagnosis.root_cause,
        diagnosis_explanation=diagnosis.explanation,
        history_context=history_context,
        policy_guidelines=policy_guidelines,
        target_content=target_content_indexed,
        tool_descriptions=available_tools_desc,
        format_instructions=analysis_parser.get_format_instructions()
    )
    
    res = llm.invoke(final_prompt)
    return analysis_parser.parse(res.content)

def regenerate_tool_args(
    llm,
    forced_tool: str,
    diagnosis: DiagnosisResult,
    query: str,
    target_content_indexed: str,
    history_context: str = "",
) -> AnalysisResult:
    """
    Regenerate tool arguments after Policy Override

    When Policy Engine forces a tool switch, original arguments may not match the new tool's schema.
    This function re-invokes LLM to generate correctly formatted arguments for the specified tool.

    Args:
        llm: LLM instance
        forced_tool: Tool name mandated by Policy
        diagnosis: Diagnosis result
        query: User query
        target_content_indexed: Target document with index
        history_context: History context

    Returns:
        AnalysisResult: Analysis result with correctly formatted arguments
    """
    # Get schema for the specified tool
    tool = registry.get_tool(forced_tool)
    if not tool:
        raise ValueError(f"Tool {forced_tool} not found in registry")

    tool_schema = tool.args_schema.schema_json()
    tool_desc = f"{tool.name}: {tool.description}"

    analysis_parser = PydanticOutputParser(pydantic_object=AnalysisResult)

    prompt = ChatPromptTemplate.from_template("""
    You are the GEO Optimization Controller. Generate the correct arguments for the SPECIFIED tool.

    ### IMPORTANT: Tool is ALREADY SELECTED
    You MUST use this tool: **{forced_tool}**
    DO NOT select a different tool. Only generate the correct arguments.

    ### Inputs
    - **Query**: {query}
    - **Root Cause Diagnosis**: {diagnosis_cause} ({diagnosis_explanation})
    - **Key Deficiency**: {key_deficiency}
    - **History**: {history_context}

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
    """)

    final_prompt = prompt.format(
        forced_tool=forced_tool,
        query=query,
        diagnosis_cause=diagnosis.root_cause,
        diagnosis_explanation=diagnosis.explanation,
        key_deficiency=diagnosis.key_deficiency,
        history_context=history_context,
        target_content=target_content_indexed[:8000],
        tool_desc=tool_desc,
        tool_schema=tool_schema,
        format_instructions=analysis_parser.get_format_instructions(),
    )

    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        try:
            res = llm.invoke(final_prompt)
            result = analysis_parser.parse(res.content)

            # Verify tool name is correct
            if result.selected_tool_name != forced_tool:
                print(f"⚠️ LLM returned wrong tool {result.selected_tool_name}, forcing to {forced_tool}")
                result.selected_tool_name = forced_tool

            return result
        except Exception as e:
            last_error = e
            print(f"Regenerate args attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                final_prompt += (
                    f"\n\nCRITICAL: Your response MUST have selected_tool_name=\"{forced_tool}\" "
                    f"and tool_arguments matching the {forced_tool} schema."
                )

    raise last_error


def analyze_failure(
    llm,
    query: str,
    indexed_target_doc: str,
    competitor_doc: str,
    memory: Optional[OptimizationMemory] = None,
    truncation_audit_summary: Optional[str] = None,
    policy_injection: str = ""
) -> tuple:
    """
    Orchestrates the OpenTelemetry-style failure analysis:
    1. Diagnose (Trace/Metric)
    2. Policy Check
    3. Act (Tool Selection)
    
    Returns:
        tuple: (AnalysisResult, DiagnosisResult) - Returns both results for Telemetry recording
    """

    # 1. Diagnose
    diagnosis = diagnose_root_cause(
        llm, query, indexed_target_doc, competitor_doc, truncation_audit_summary
    )
    print(f"🧐 Diagnosis: {diagnosis.root_cause} - {diagnosis.key_deficiency}")
    
    # 2. Prepare Context
    history_context = ""
    if memory and memory.modifications:
        history_context = memory.get_history_summary()
        
    tool_descs = "\n".join([f"{t.name}: {t.description} - Args: {t.args_schema.schema_json()}" for t in registry.get_all_tools()])
    
    # 3. Select Tool (with Policy Injection)
    analysis = select_tool_strategy(
        llm, 
        query, 
        diagnosis, 
        indexed_target_doc, 
        tool_descs, 
        history_context,
        memory,
        policy_injection=policy_injection
    )
    
    return analysis, diagnosis

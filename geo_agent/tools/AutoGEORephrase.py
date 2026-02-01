"""
AutoGEO Rephrase Tool - Meta-Tool for comprehensive document rewriting.

This tool integrates the AutoGEO paper's methodology into the GEO Agent's tool system,
providing access to 9 validated rule sets (3 datasets × 3 engines) for maximum visibility
optimization in generative search engines.
"""
from typing import Literal, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from geo_agent.config import get_llm_from_config
from .registry import registry
from .utils import build_context_section, HTMLFragmentProcessor


# ============================================================================
# AutoGEO Rule Sets - 9 combinations (3 datasets × 3 engines)
# ============================================================================

AUTOGEO_RULES = {
    # Researchy-GEO (research queries) + Gemini
    ("researchy", "gemini"): [
        "Attribute all factual claims to credible, authoritative sources with clear citations.",
        "Cover the topic comprehensively, addressing all key aspects and sub-topics.",
        "Ensure information is factually accurate and verifiable.",
        "Focus exclusively on the topic, eliminating irrelevant information, navigational links, and advertisements.",
        "Maintain a neutral, objective tone, avoiding promotional language, personal opinions, and bias.",
        "Maintain high-quality writing, free from grammatical errors, typos, and formatting issues.",
        "Present a balanced perspective on complex topics, acknowledging multiple significant viewpoints or counter-arguments.",
        "Present information as a self-contained unit, not requiring external links for core understanding.",
        "Provide clear, specific, and actionable steps.",
        "Provide explanatory depth by clarifying underlying causes, mechanisms, and context ('how' and 'why').",
        "State the key conclusion at the beginning of the document.",
        "Structure content logically with clear headings, lists, and paragraphs to ensure a cohesive flow.",
        "Substantiate claims with specific, concrete details like data, statistics, or named examples.",
        "Use clear and concise language, avoiding jargon, ambiguity, and verbosity.",
        "Use current information, reflecting the latest state of knowledge."
    ],

    # GEO-Bench (research queries) + Gemini
    ("geo_bench", "gemini"): [
        "Ensure all information is factually accurate and verifiable, citing credible sources.",
        "Ensure information is current and up-to-date, especially for time-sensitive topics.",
        "Ensure the document is self-contained and comprehensive, providing all necessary context and sub-topic information.",
        "Explain the underlying mechanisms and principles (the 'why' and 'how'), not just surface-level facts.",
        "Maintain a singular focus on the core topic, excluding tangential information, promotional content, and document 'noise' (e.g., navigation, ads).",
        "Organize content with a clear, logical hierarchy, using elements like headings, lists, and tables.",
        "Present a balanced and objective view on debatable topics, including multiple significant perspectives.",
        "Provide specific, actionable guidance, such as step-by-step instructions, for procedural topics.",
        "State the primary conclusion directly at the beginning of the document.",
        "Use clear and unambiguous language, defining technical terms, acronyms, and jargon upon first use.",
        "Use specific, concrete details and examples instead of abstract generalizations.",
        "Write concisely, eliminating verbose language, redundancy, and filler content."
    ],

    # E-commerce (commercial queries) + Gemini
    ("ecommerce", "gemini"): [
        "Ensure all information is factually accurate, verifiable, and current for the topic.",
        "Establish credibility by citing authoritative sources, providing evidence, or demonstrating clear expertise.",
        "Justify recommendations and claims with clear reasoning, context, or comparative analysis like pros and cons.",
        "Organize content with a clear, logical structure using elements like headings, lists, and tables to facilitate scanning and parsing.",
        "Present information objectively, avoiding promotional bias and including balanced perspectives where applicable.",
        "Provide actionable information, such as step-by-step instructions or clear recommendations.",
        "Provide specific, verifiable details such as names, model numbers, technical specifications, and quantifiable data.",
        "Structure content into modular, self-contained units, such as distinct paragraphs or list items for each concept.",
        "Use clear, simple, and unambiguous language, defining any necessary technical terms or jargon.",
        "Write concisely, eliminating verbose language, filler content, and unnecessary repetition."
    ],

    # Researchy-GEO (research queries) + GPT
    ("researchy", "gpt"): [
        "Attribute all claims to specific, credible, and authoritative sources.",
        "Create a self-contained document, free from non-informational content like advertisements, navigation, or paywalls.",
        "Ensure all content is strictly relevant to the core topic, excluding tangential or unrelated information.",
        "Ensure all information is factually accurate, verifiable, and internally consistent.",
        "Ensure content is fully accessible without requiring logins, subscriptions, or payments.",
        "Ensure information is current and up-to-date, especially for time-sensitive topics.",
        "Explain underlying mechanisms and causal relationships (the 'how' and 'why'), not just descriptive facts.",
        "Maintain a neutral and objective tone, prioritizing factual information over subjective opinions or biased language.",
        "Maintain a purely informational purpose, avoiding promotional, persuasive, or interactive content.",
        "Organize content with a clear, logical structure, using elements like headings and lists to improve readability.",
        "Present a balanced perspective on complex topics by including multiple relevant viewpoints or counterarguments.",
        "Present information with a cohesive, logical flow, avoiding fragmented or contradictory statements.",
        "Provide comprehensive coverage of the topic, addressing its key facets, nuances, and relevant context.",
        "Provide specific, actionable guidance when the topic involves a task or problem-solving.",
        "State the key conclusion directly at the beginning of the document.",
        "Substantiate claims with specific evidence, such as quantifiable data or concrete examples.",
        "Use clear, concise, and unambiguous language, defining essential jargon and eliminating filler content."
    ],

    # GEO-Bench (research queries) + GPT
    ("geo_bench", "gpt"): [
        "Address the topic comprehensively, covering all essential sub-topics and necessary context.",
        "Define essential terms, acronyms, and jargon upon their first use.",
        "Ensure all factual information is accurate, verifiable, and internally consistent.",
        "Ensure content is free from illegal, unethical, or harmful information.",
        "Ensure each document is self-contained, providing all necessary information on the topic without requiring external links.",
        "Explain the 'why' and 'how' behind facts, clarifying underlying principles and mechanisms.",
        "Explicitly differentiate between similar or easily confused concepts.",
        "For complex or debatable subjects, present multiple significant viewpoints in a balanced way.",
        "For procedural content, provide clear, numbered, step-by-step instructions.",
        "For time-sensitive topics, ensure information is current and clearly display its publication or last-updated date.",
        "Maintain a neutral, objective tone, clearly distinguishing facts from opinions.",
        "Maintain a singular focus on the core topic, excluding tangential or promotional content.",
        "Organize content with a clear, logical hierarchy using headings, lists, and tables.",
        "State the primary conclusion at the beginning of the document.",
        "Structure content into atomic units, where each paragraph or section addresses a single idea.",
        "Use clear, simple, and unambiguous language.",
        "Use concrete examples, analogies, or case studies to illustrate complex concepts.",
        "Use specific, concrete details like names, dates, and statistics instead of generalizations.",
        "Write concisely, eliminating repetition, filler words, and verbose phrasing."
    ],

    # E-commerce (commercial queries) + GPT
    ("ecommerce", "gpt"): [
        "Be complete and thorough, covering all key aspects and a sufficient range of options.",
        "Clearly define the document's scope, especially for broad or ambiguous topics.",
        "Ensure all factual information is accurate, verifiable, and objective.",
        "Ensure information is up-to-date, clearly indicating its publication or last-updated date.",
        "Ensure the document is a complete, self-contained unit, not truncated or missing essential information.",
        "Establish credibility by citing authoritative sources or explaining the methodology for arriving at conclusions.",
        "Maintain a neutral tone, free from bias, promotional language, and unsubstantiated claims.",
        "Organize content logically with a clear, hierarchical structure using elements like headings, lists, and tables for easy parsing.",
        "Present information concisely, eliminating verbose language, filler words, and unnecessary introductions.",
        "Prioritize the most critical information by placing it at the beginning of the document or relevant section.",
        "Provide actionable content, such as step-by-step instructions or clear recommendations.",
        "Provide context and explain the reasoning behind recommendations, conclusions, or complex information.",
        "Structure data in a way that allows for direct evaluation, such as in a table or a pros-and-cons list.",
        "Use simple, direct, and unambiguous language, defining any necessary technical jargon.",
        "Use specific, quantifiable details like names, metrics, and technical specifications instead of vague generalizations."
    ],

    # Researchy-GEO (research queries) + Claude
    ("researchy", "claude"): [
        "Cover the topic comprehensively by addressing all its key facets and relevant sub-topics.",
        "Dedicate each paragraph or self-contained section to a single, distinct idea.",
        "Ensure a cohesive narrative flow where ideas connect logically rather than appearing as disconnected facts.",
        "Ensure all information is factually accurate, internally consistent, and up-to-date.",
        "Ensure the document is self-contained, providing all necessary context without requiring readers to follow external links.",
        "Ensure the full text is programmatically accessible, without requiring logins, paywalls, or user interaction.",
        "Focus exclusively on a single topic, removing all tangential information, advertisements, and navigational elements.",
        "Illustrate concepts and support arguments with specific details, concrete examples, or data.",
        "Maintain a neutral, objective tone, clearly distinguishing facts from opinions and avoiding biased or promotional language.",
        "Organize content with a clear, logical hierarchy using headings, lists, or tables to facilitate machine parsing.",
        "Present a balanced perspective on debatable topics by acknowledging multiple significant viewpoints or counterarguments.",
        "Provide clear, actionable steps or practical guidance for procedural topics.",
        "Provide explanatory depth by detailing the underlying mechanisms, causes, and effects ('how' and 'why').",
        "State the primary conclusion directly at the beginning of the document.",
        "Substantiate all claims with citations to credible, authoritative sources.",
        "Use clear and unambiguous language, defining specialized or technical terms upon their first use.",
        "Write concisely, eliminating repetitive phrasing, filler content, and unnecessary verbosity."
    ],

    # GEO-Bench (research queries) + Claude
    ("geo_bench", "claude"): [
        "Cite authoritative sources to support claims and establish credibility.",
        "Cover the topic comprehensively, providing depth by explaining the underlying 'why' and 'how'.",
        "Ensure all information is factually accurate, verifiable, and internally consistent.",
        "Ensure each document is self-contained and can be understood without external context.",
        "Focus on a single topic, writing concisely and eliminating irrelevant or repetitive content.",
        "For task-oriented topics, provide actionable guidance like step-by-step instructions.",
        "Indicate the timeliness of information with clear publication or revision dates.",
        "Maintain a neutral, objective tone, prioritizing facts over opinions or promotional language.",
        "Present multiple perspectives and counterarguments for complex or debatable topics.",
        "Provide specific details, such as names, dates, statistics, and concrete examples, to support claims and illustrate concepts.",
        "Segment content into discrete units, where each paragraph or list item addresses a single idea.",
        "State the key conclusion at the beginning of the document.",
        "Use clear structural elements like headings, lists, and tables to organize content logically.",
        "Use clear, unambiguous language, and define technical terms or acronyms on their first use."
    ],

    # E-commerce (commercial queries) + Claude
    ("ecommerce", "claude"): [
        "Eliminate all tangential or promotional information.",
        "Ensure all information is factually accurate and verifiable, supporting claims with citations to authoritative sources.",
        "Ensure core content is directly accessible, without requiring logins, paywalls, or complex navigation.",
        "Keep information current for time-sensitive topics and clearly state its timeliness.",
        "Maintain an objective, neutral tone and present a balanced perspective, including relevant pros and cons or alternative viewpoints where applicable.",
        "Maintain internal consistency in terminology, formatting, and data presentation, especially for comparable items.",
        "Organize content using a clear, logical, and consistent structure with elements like headings, lists, and tables to facilitate automated parsing.",
        "Provide actionable content, such as step-by-step instructions or direct recommendations.",
        "Provide context or rationale to explain the reasoning behind data, recommendations, or claims.",
        "Structure content into discrete, self-contained units, with each paragraph or section addressing a single concept.",
        "The document should provide the complete core information, without requiring navigation to external links for essential information.",
        "Use specific, quantifiable details like names, model numbers, and metrics instead of vague generalizations.",
        "Write with clarity and conciseness, using simple, direct language and eliminating unnecessary jargon, repetition, and filler."
    ],
}


class AutoGEORephraseInput(BaseModel):
    """Input schema for the AutoGEO Rephrase meta-tool."""
    dataset_type: Literal["researchy", "ecommerce", "geo_bench"] = Field(
        default="researchy",
        description="Dataset type to select appropriate rule set: 'researchy' for research queries, 'ecommerce' for commercial queries, 'geo_bench' for benchmark queries."
    )
    engine_llm: Literal["gemini", "gpt", "claude"] = Field(
        default="gemini",
        description="Target generative engine LLM: 'gemini', 'gpt', or 'claude'. Rules are optimized for each engine."
    )
    target_content: str = Field(
        ...,
        description="The HTML/text content to be comprehensively rewritten."
    )
    context_before: str = Field(
        default="",
        description="Read-only context before the target content."
    )
    context_after: str = Field(
        default="",
        description="Read-only context after the target content."
    )
    core_idea: str = Field(
        default="",
        description="The core topic of the document that must be preserved."
    )
    previous_modifications: str = Field(
        default="",
        description="Summary of previous modifications to preserve."
    )
    custom_rule_path: Optional[str] = Field(
        default=None,
        description="Optional path to a custom rule JSON file. If provided, overrides the default rules."
    )


def _load_custom_rules(rule_path: str) -> Optional[list]:
    """Load rules from a custom JSON file."""
    import json
    try:
        with open(rule_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'filtered_rules' in data:
                return data['filtered_rules']
            if isinstance(data, list):
                return data
    except Exception as e:
        print(f"Warning: Failed to load custom rules from {rule_path}: {e}")
    return None


def autogeo_rephrase(
    target_content: str,
    dataset_type: str = "researchy",
    engine_llm: str = "gemini",
    context_before: str = "",
    context_after: str = "",
    core_idea: str = "",
    previous_modifications: str = "",
    custom_rule_path: Optional[str] = None
) -> str:
    """
    Comprehensively rewrite document using AutoGEO methodology with validated rule sets.

    This meta-tool applies 12-19 optimization rules derived from the AutoGEO paper,
    performing holistic document transformation for maximum visibility in generative search.
    """
    llm = get_llm_from_config('geo_agent/config.yaml')

    # Select rules: custom > default based on dataset/engine combo
    rules = None
    if custom_rule_path:
        rules = _load_custom_rules(custom_rule_path)

    if not rules:
        rules = AUTOGEO_RULES.get((dataset_type, engine_llm))
        if not rules:
            # Fallback to researchy + gemini
            rules = AUTOGEO_RULES[("researchy", "gemini")]

    # Format rules as string
    rules_string = "\n".join(f"  {i+1}. {rule}" for i, rule in enumerate(rules))

    # Build context section
    context_section = build_context_section(context_before, context_after)

    # Build history section
    history_section = ""
    if previous_modifications:
        history_section = f"""
⚠️ PREVIOUS MODIFICATIONS (MUST PRESERVE):
{previous_modifications}
"""

    prompt = ChatPromptTemplate.from_template("""
You are an expert content optimizer implementing the **AutoGEO methodology** for generative search engine optimization.

Your task is to **comprehensively rewrite** the document to maximize its visibility and citation probability in generative search engines (like Gemini, ChatGPT, Claude).

🎯 CORE TOPIC: {core_idea}
{context_section}
{history_section}

## AutoGEO Quality Guidelines
These {rule_count} rules are derived from the AutoGEO paper and validated through experiments on {dataset_desc} for {engine_desc}:

{rules_string}

## Key Transformation Goals
Apply these transformations holistically:

1. **Credibility Enhancement**
   - Add citations to authoritative sources
   - Use specific data, statistics, and named examples
   - Establish expertise through precise, verifiable claims

2. **Structural Optimization**
   - Use clear headings and logical hierarchy
   - Employ lists for scannable content
   - Create self-contained, modular sections

3. **Clarity Improvement**
   - Eliminate jargon or define technical terms
   - Use concise, direct language
   - Remove redundancy and filler content

4. **Completeness Assurance**
   - Cover all key aspects of the topic
   - Provide the "how" and "why", not just "what"
   - Include actionable guidance where appropriate

5. **Engagement Optimization**
   - State key conclusions upfront (BLUF principle)
   - Maintain neutral, objective tone
   - Present balanced perspectives on complex topics

## Critical Constraints
⚠️ CORE FOCUS: The document MUST remain about "{core_idea}" - DO NOT drift to unrelated topics.
⚠️ NO INFORMATION LOSS: Preserve all existing facts while enhancing presentation.
⚠️ FORMAT PRESERVATION: Maintain HTML/text format and essential structural elements.
⚠️ PRESERVE MODIFICATIONS: Honor all previous modifications.

=== TARGET CONTENT ===
{target_content}
=== END TARGET CONTENT ===

## Output Format
First output the comprehensively rewritten content, then provide a modification summary.

[Rewritten content with all AutoGEO rules applied...]

---MODIFICATION_SUMMARY---
- [Rule applied: ...]
- [Enhancement made: ...]
- [Structural change: ...]
""")

    # Descriptions for the prompt
    dataset_descriptions = {
        "researchy": "research/informational queries",
        "ecommerce": "commercial/product queries",
        "geo_bench": "academic benchmark queries"
    }
    engine_descriptions = {
        "gemini": "Google Gemini",
        "gpt": "OpenAI GPT",
        "claude": "Anthropic Claude"
    }

    response = llm.invoke(prompt.format(
        target_content=target_content,
        context_section=context_section,
        core_idea=core_idea or "the document's main topic",
        history_section=history_section,
        rules_string=rules_string,
        rule_count=len(rules),
        dataset_desc=dataset_descriptions.get(dataset_type, "general queries"),
        engine_desc=engine_descriptions.get(engine_llm, "generative AI")
    ))

    return response.content


# Register the tool with a prominent description
registry.register(
    autogeo_rephrase,
    AutoGEORephraseInput,
    name="autogeo_rephrase",
    description="""[Academic Research-Based] Comprehensive document rewriting using AutoGEO methodology.

META-TOOL applying 12-19 validated optimization rules for maximum visibility in generative search engines.

WHEN TO USE:
- Multiple diagnosis issues (LOW_INFO_DENSITY + STRUCTURAL_WEAKNESS + TRUST_CREDIBILITY)
- Previous targeted tools failed to achieve citation
- Document needs holistic improvement across multiple dimensions
- Quality problems are systemic rather than localized

CAPABILITIES:
- 9 rule sets optimized for 3 datasets (researchy/ecommerce/geo_bench) × 3 engines (gemini/gpt/claude)
- Comprehensive transformation: credibility, structure, clarity, completeness
- Preserves core topic while enhancing all quality dimensions
- Academic paper-validated optimization strategies

DIFFERENTIATION FROM OTHER TOOLS:
- entity_injection: Only adds missing facts → autogeo_rephrase: Full rewrite
- structure_optimization: Only improves layout → autogeo_rephrase: Structure + content + credibility
- intent_realignment: Only aligns with query → autogeo_rephrase: Alignment + quality + authority"""
)

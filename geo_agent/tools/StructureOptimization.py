from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from geo_agent.config import get_llm_from_config
from .registry import registry
from .utils import build_context_section, PRESERVATION_RULES


class StructureOptimizationInput(BaseModel):
    target_content: str = Field(..., description="Unstructured HTML chunk (wall of text).")
    context_before: str = Field("", description="Read-only context before.")
    context_after: str = Field("", description="Read-only context after.")
    core_idea: str = Field("", description="The core topic of the document that must be preserved.")
    previous_modifications: str = Field("", description="Summary of previous modifications.")

def optimize_structure(target_content: str, context_before: str = "", context_after: str = "", core_idea: str = "", previous_modifications: str = "") -> str:
    """Injects semantic HTML tags (H2, UL, Strong) to structure the text."""
    llm = get_llm_from_config('geo_agent/config.yaml')
    context_section = build_context_section(context_before, context_after)

    # Build history section
    history_section = ""
    if previous_modifications:
        history_section = f"""
⚠️ PREVIOUS MODIFICATIONS (MUST PRESERVE):
{previous_modifications}
"""

    prompt = ChatPromptTemplate.from_template("""
You are an HTML Semantic Architect. Transform the unstructured text into parser-friendly, structured HTML.

{context_section}

Core idea: {core_idea}
{history_section}

{preservation_rules}

SPECIFIC INSTRUCTIONS:
1. **Semantic Hierarchy**: Insert `<h3>` or `<h4>` tags (based on context depth) to break up long text blocks.
2. **List Conversion**: Detect sentences that list items (e.g., "We have A, B, and C") and convert them into `<ul><li>` or `<ol><li>`.
3. **Emphasis**: Wrap key entities (defined in the text) with `<strong>` or `<em>`.
4. **NO TEXT CHANGES**: You are strictly forbidden from changing the wording, tone, or length of the text. ONLY add HTML tags.

=== TARGET CONTENT (HTML Fragment) ===
{target_content}
=== END TARGET CONTENT ===

OUTPUT FORMAT:
[Structured HTML content...]

---MODIFICATION_SUMMARY---
- [Change 1]
""")

    response = llm.invoke(prompt.format(
        target_content=target_content, 
        context_section=context_section,
        preservation_rules=PRESERVATION_RULES,
        core_idea=core_idea,
        history_section=history_section
    ))
    return response.content

registry.register(optimize_structure, StructureOptimizationInput, name="structure_optimization", description="Adds semantic HTML tags (H2-H6, UL, Strong) to unstructured text without changing words.")
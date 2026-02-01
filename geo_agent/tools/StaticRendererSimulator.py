from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from geo_agent.config import get_llm_from_config
from .registry import registry
from .utils import build_context_section, PRESERVATION_RULES


class StaticRenderingInput(BaseModel):
    target_content: str = Field(..., description="HTML chunk containing JS code or placeholders.")
    context_before: str = Field("", description="Read-only context before.")
    context_after: str = Field("", description="Read-only context after.")
    core_idea: str = Field("", description="The core topic of the document that must be preserved.")
    previous_modifications: str = Field("", description="Summary of previous modifications.")

def simulate_static_render(target_content: str, context_before: str = "", context_after: str = "", core_idea: str = "", previous_modifications: str = "") -> str:
    """Simulates Server-Side Rendering (SSR) by converting JS/Placeholders to static HTML."""
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
You are a Server-Side Rendering (SSR) Simulator.
The input content contains JavaScript code, JSON blobs, or "Loading..." placeholders that hide the actual content.
Your task is to "render" this into static, readable HTML text based on what the JS code implies.

{context_section}

Core idea: {core_idea}
{history_section}

{preservation_rules}

SPECIFIC INSTRUCTIONS:
1. Look at the JS code or JSON data in the input. 
2. Extract the actual human-readable content (text, prices, items) from the code logic.
3. Output the content as plain, static HTML tags (`<p>`, `<ul>`, `<div>`).
4. **REMOVE** the `<script>` tags and raw JSON.
5. If the JS implies fetching external data that is NOT present in the chunk, place a placeholder: `` but do not hallucinate data.

=== TARGET CONTENT (Raw/JS-heavy Chunk) ===
{target_content}
=== END TARGET CONTENT ===

OUTPUT FORMAT:
[Static HTML Content...]

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

registry.register(simulate_static_render, StaticRenderingInput, name="static_rendering", description="Converts JS-heavy or dynamic chunks into static, parser-readable HTML.")
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from geo_agent.config import get_llm_from_config
from .registry import registry
from .utils import build_context_section, PRESERVATION_RULES

class NoiseIsolationInput(BaseModel):
    target_content: str = Field(..., description="HTML chunk containing mixed content and noise.")
    context_before: str = Field("", description="Read-only context before.")
    context_after: str = Field("", description="Read-only context after.")
    core_idea: str = Field("", description="The core topic of the document that must be preserved.")
    previous_modifications: str = Field("", description="Summary of previous modifications.")

def isolate_noise(target_content: str, context_before: str = "", context_after: str = "", core_idea: str = "", previous_modifications: str = "", config_path: str = "geo_agent/config.yaml") -> str:
    """Wraps non-semantic content (nav, ads) in <aside> or <nav> tags."""
    llm = get_llm_from_config(config_path)
    context_section = build_context_section(context_before, context_after)

    # Build history section
    history_section = ""
    if previous_modifications:
        history_section = f"""
⚠️ PREVIOUS MODIFICATIONS (MUST PRESERVE):
{previous_modifications}
"""

    prompt = ChatPromptTemplate.from_template("""
You are an HTML Semantic Cleaner. Your job is to tell search engine parsers what is NOISE and what is CONTENT.

{context_section}

Core idea: {core_idea}
{history_section}

{preservation_rules}

SPECIFIC INSTRUCTIONS:
1. Analyze the target content. Identify elements that are likely:
   - Navigation links
   - Cookie warnings
   - Sidebar ads / "Read More" links
   - Copyright footers
2. Wrap these elements in `<aside>`, `<nav>`, or `<footer class="geo-noise">` tags.
3. Wrap the actual high-value semantic content in `<article>` or `<section>`.
4. **DO NOT DELETE** the noise (as it might be functionally needed for the webpage UI), just wrap it semantically so parsers can de-prioritize it.

=== TARGET CONTENT (HTML Fragment) ===
{target_content}
=== END TARGET CONTENT ===

OUTPUT FORMAT:
[Semantically Wrapped HTML...]

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

registry.register(isolate_noise, NoiseIsolationInput, name="noise_isolation", description="Wraps boilerplate/noise in semantic tags (aside, nav) to help parsers focus on main content.")
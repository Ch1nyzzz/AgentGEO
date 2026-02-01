from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from geo_agent.config import get_llm_from_config
from .registry import registry
from .utils import build_context_section, PRESERVATION_RULES, HTMLFragmentProcessor

try:
    from core.memory import parse_tool_output
except ImportError:
    try:
        from geo_agent.core.memory import parse_tool_output
    except ImportError:
        def parse_tool_output(text):
            if "---MODIFICATION_SUMMARY---" in text:
                parts = text.split("---MODIFICATION_SUMMARY---", 1)
                return parts[0].strip(), parts[1].strip()
            return text, ""


class BlufOptimizationInput(BaseModel):
    key_takeaway: str = Field(..., description="The specific answer/insight that must be foregrounded.")
    target_content: str = Field(..., description="The HTML chunk containing the buried info.")
    context_before: str = Field("", description="Read-only context before.")
    context_after: str = Field("", description="Read-only context after.")
    core_idea: str = Field("", description="The core topic of the document that must be preserved.")
    previous_modifications: str = Field("", description="Summary of previous modifications.")

def optimize_bluf(key_takeaway: str, target_content: str, context_before: str = "", context_after: str = "", core_idea: str = "", previous_modifications: str = "") -> str:
    """Adds a 'Bottom Line Up Front' (BLUF) summary block at the top of the chunk."""
    llm = get_llm_from_config('geo_agent/config.yaml')
    context_section = build_context_section(context_before, context_after)
    
    # Init Processor
    processor = HTMLFragmentProcessor(target_content)

    # Build history section
    history_section = ""
    if previous_modifications:
        history_section = f"""
⚠️ PREVIOUS MODIFICATIONS (MUST PRESERVE):
{previous_modifications}
"""

    prompt = ChatPromptTemplate.from_template("""
You are a UX Writer for Information Retrieval. Surface the key takeaway immediately using the BLUF (Bottom Line Up Front) principle.

Key Takeaway to Surface: "{key_takeaway}"

{context_section}
                                              
Core Idea: {core_idea}
{history_section}

SPECIFIC INSTRUCTIONS:
1. Generate the HTML content for a summary box.
2. Summarize the "{key_takeaway}" in one direct, factual sentence using `<strong>`.
3. Do NOT output the existing content, ONLY the new summary content.

OUTPUT FORMAT:
[HTML Content for the BLUF Summary Div ONLY]

---MODIFICATION_SUMMARY---
- [Change 1]
""")

    response = llm.invoke(prompt.format(
        key_takeaway=key_takeaway, 
        target_content=target_content, 
        context_section=context_section,
        preservation_rules=PRESERVATION_RULES,
        core_idea=core_idea,
        history_section=history_section
    ))
    
    # Parse LLM output (New Content | Summary)
    new_bluf_content, mod_summary = parse_tool_output(response.content)
    
    # Inject into HTML via BS4
    processor.prepend_container(new_bluf_content, tag="div", attrs={"class": "geo-summary-box"})
    
    # Construct final output (HTML + Summary)
    final_html = processor.to_html()
    return f"{final_html}\n\n---MODIFICATION_SUMMARY---\n{mod_summary}"

registry.register(optimize_bluf, BlufOptimizationInput, name="bluf_optimization", description="Adds a summary block at the top of the content to foreground buried information.")
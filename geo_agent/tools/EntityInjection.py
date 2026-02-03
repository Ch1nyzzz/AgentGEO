from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from geo_agent.config import get_llm_from_config
from .registry import registry
from .utils import build_context_section, PRESERVATION_RULES, HTMLFragmentProcessor


class EntityInjectionInput(BaseModel):
    missing_entities: str = Field(..., description="The specific facts, numbers, or attributes missing from the content (e.g., 'Price: $299', 'Battery: 5000mAh').")
    target_content: str = Field(..., description="The HTML chunk where the entities should be contextually inserted.")
    context_before: str = Field("", description="Read-only context before the target content.")
    context_after: str = Field("", description="Read-only context after the target content.")
    core_idea: str = Field("", description="The core topic of the document that must be preserved.")
    previous_modifications: str = Field("", description="Summary of previous modifications.")

def inject_entities(missing_entities: str, target_content: str, context_before: str = "", context_after: str = "", core_idea: str = "", previous_modifications: str = "", config_path: str = "geo_agent/config.yaml") -> str:
    """Weaves missing specific entities/facts naturally into the HTML content."""
    llm = get_llm_from_config(config_path)
    
    # 1. Preprocessing: ensure target_content is a valid HTML fragment
    # EntityInjection needs context to decide insertion position, so we pass HTML to LLM
    # But we use Processor to ensure final output normalization
    processor = HTMLFragmentProcessor(target_content)
    # Get plain text for LLM reference (optional, but sometimes LLM can better locate with HTML structure)
    # Here we pass HTML directly to LLM because we need to preserve original tag structure
    
    context_section = build_context_section(context_before, context_after)

    # Build history section
    history_section = ""
    if previous_modifications:
        history_section = f"""
⚠️ PREVIOUS MODIFICATIONS (MUST PRESERVE):
{previous_modifications}
"""

    prompt = ChatPromptTemplate.from_template("""
You are a Content Enricher and HTML Editor. 
Your task is to perform a "surgical injection" of missing information into the provided HTML fragment.

Missing Entities/Facts to Inject: "{missing_entities}"

{context_section}
                                              
Core idea: {core_idea}
{history_section}

{preservation_rules}

SPECIFIC INSTRUCTIONS:
1. **Locate the Best Spot**: Analyze the HTML content to find the most semantically relevant place.
2. **Rewriting**: Rewrite the relevant sentence/paragraph to naturally include the new fact. 
3. **Semantic Highlighting**: Wrap the injected entity in `<strong>` tags.
4. **Preserve Structure**: Do NOT remove existing links (`<a href...>`) or formatting unless necessary for the rewrite.
5. **Output**: Return the FULL modified HTML chunk.

=== TARGET CONTENT (HTML Fragment) ===
{target_content}
=== END TARGET CONTENT ===

OUTPUT FORMAT:
first output the enhanced HTML content, then the summary.

[Enhanced HTML content...]

---MODIFICATION_SUMMARY---
- [Inserted '{missing_entities}']
""")

    response = llm.invoke(prompt.format(
        missing_entities=missing_entities,
        target_content=target_content, 
        context_section=context_section,
        core_idea=core_idea,
        preservation_rules=PRESERVATION_RULES,
        history_section=history_section
    ))
    
    # 2. Parse & validate
    # modified_html, mod_summary = parse_tool_output(response.content)

    # Here we return the LLM modified HTML directly because it's a "minimally invasive" operation,
    # Python cannot easily locate the insertion point.
    # But we can optionally use Processor to clean and ensure it's properly closed
    # processor_out = HTMLFragmentProcessor(modified_html)
    # final_html = processor_out.to_html()
    
    return response.content #f"{modified_html}\n\n---MODIFICATION_SUMMARY---\n{mod_summary}"

registry.register(inject_entities, EntityInjectionInput, name="entity_injection", description="Injects missing specific facts or entities into the HTML content naturally, using strong tags for emphasis.")
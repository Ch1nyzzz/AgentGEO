from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from geo_agent.config import get_llm_from_config
from .registry import registry
from .utils import build_context_section, PRESERVATION_RULES


class DataSerializationInput(BaseModel):
    target_content: str = Field(..., description="HTML chunk containing narrative data.")
    context_before: str = Field("", description="Read-only context before.")
    context_after: str = Field("", description="Read-only context after.")
    core_idea: str = Field("", description="The core topic of the document that must be preserved.")
    previous_modifications: str = Field("", description="Summary of previous modifications.")

def serialize_data(target_content: str, context_before: str = "", context_after: str = "", core_idea: str = "", previous_modifications: str = "", config_path: str = "geo_agent/config.yaml") -> str:
    """Converts narrative data (sentences) into HTML tables or definition lists."""
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
You are a Data Structuring Agent. Convert narrative descriptions of data into structured HTML Tables.

{context_section}

Core idea: {core_idea}
{history_section}

{preservation_rules}

SPECIFIC INSTRUCTIONS:
1. Identify repeating data patterns in the text (e.g., "Product A costs $10. Product B costs $20.").
2. Create an HTML `<table>` with appropriate `<thead>` headers to represent this data.
3. **CRITICAL**: Do NOT delete the original narrative text completely if it contains nuance/sentiment that fits poorly in a table. Instead, keep the text as a summary *after* the table, or integrate the nuance into a "Notes" column.
4. Ensure every number and entity from the text exists in the table.

=== TARGET CONTENT (HTML Fragment) ===
{target_content}
=== END TARGET CONTENT ===

OUTPUT FORMAT:
[HTML Content with Table...]

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

registry.register(serialize_data, DataSerializationInput, name="data_serialization", description="Converts narrative text containing data points into structured HTML tables.")
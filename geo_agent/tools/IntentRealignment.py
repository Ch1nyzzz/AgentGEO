from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from geo_agent.config import get_llm_from_config
from .registry import registry
from .utils import build_context_section, PRESERVATION_RULES

class IntentRealignmentInput(BaseModel):
    user_query: str = Field(..., description="The target user query we want to rank for.")
    target_content: str = Field(..., description="The specific HTML chunk to be re-aligned.")
    context_before: str = Field("", description="Read-only context before.")
    context_after: str = Field("", description="Read-only context after.")
    core_idea: str = Field("", description="The core topic of the document that must be preserved.")
    previous_modifications: str = Field("", description="Summary of previous modifications to preserve.")

    

def realign_intent(user_query: str, target_content: str, context_before: str = "", context_after: str = "", core_idea: str = "", previous_modifications: str = "") -> str:
    """Rewrites the content start to directly address the user intent."""
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
You are an expert SEO Content Editor. Your goal is to make the content explicitly answer the User Query immediately, without losing original depth.

User Query: "{user_query}"

{context_section}

Core idea: {core_idea}
{history_section}

{preservation_rules}

SPECIFIC INSTRUCTIONS:
1. Analyze the intent of the query (Informational/Navigational/Commercial).
2. Rewrite the specific TARGET CONTENT to address this intent in the first sentence.
3. Remove "fluff" or vague introductions (e.g., "In today's world...").
4. Ensure keywords related to the query appear naturally.
5. **CRITICAL**: If the original content contains specific details (dates, names, specs) not related to the query, MOVE them to the end of the chunk, DO NOT DELETE THEM.

=== TARGET CONTENT (HTML Fragment) ===
{target_content}
=== END TARGET CONTENT ===

OUTPUT FORMAT:
First output the enhanced HTML content, then the summary.

[Enhanced HTML content...]

---MODIFICATION_SUMMARY---
- [Change 1]
""")

    response = llm.invoke(prompt.format(
        user_query=user_query, 
        target_content=target_content, 
        context_section=context_section,
        preservation_rules=PRESERVATION_RULES,
        core_idea=core_idea,
        history_section=history_section
    ))
    return response.content

registry.register(realign_intent, IntentRealignmentInput, name="intent_realignment", description="Re-aligns content to directly answer a user query while preserving details.")
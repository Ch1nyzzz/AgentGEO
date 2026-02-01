
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from geo_agent.config import get_llm_from_config
from .registry import registry
from typing import Literal


PERSUASION_STRATEGIES = {
    "authoritative_tone": """
- Use professional terminology and industry-standard expressions
- Cite specific data, research findings, or expert opinions where available
- Adopt objective, confident statement style
- Replace vague words like "maybe", "perhaps" with "research shows", "data indicates"
- Add credibility markers: credentials, experience, proven track record
""",
    "counter_argument": """
- First acknowledge the validity of opposing viewpoints
- Then refute with stronger evidence and reasoning
- Use structures like "While... however..." or "Although... nevertheless..."
- Demonstrate comprehensive thinking to increase credibility
- Address potential objections proactively
""",
    "emotional_hook": """
- Open with a story, scenario, or compelling question to capture attention
- Connect to the reader's pain points, desires, or aspirations
- Use second person "you" to increase immersion and personal relevance
- Add emotionally resonant words at key argument points
- Create vivid imagery that readers can relate to
""",
    "social_proof": """
- Emphasize "most people", "expert consensus", "industry standard"
- Reference user reviews, case studies, success stories
- Use numbers for credibility (e.g., "90% of users", "trusted by 10,000+")
- Highlight endorsements, awards, or recognition
- Show that others have made the same choice successfully
""",
    "scarcity_urgency": """
- Emphasize timeliness, scarcity, or uniqueness
- Use words like "only", "first", "exclusive", "limited"
- Imply the cost of missing out (FOMO)
- Highlight what makes this opportunity rare or time-sensitive
- Create a sense of immediate value
""",
    "logical_structure": """
- Present clear cause-and-effect relationships
- Use numbered lists and step-by-step reasoning
- Provide concrete evidence for each claim
- Build arguments progressively from premise to conclusion
- Eliminate logical gaps and strengthen transitions
"""
}
class PersuasionInput(BaseModel):
    strategy: Literal[
        "authoritative_tone",
        "counter_argument",
        "emotional_hook",
        "social_proof",
        "scarcity_urgency",
        "logical_structure"
    ] = Field(..., description="Persuasion strategy: authoritative_tone, counter_argument, emotional_hook, social_proof, scarcity_urgency, logical_structure")
    target_content: str = Field(..., description="The current content to enhance.")
    core_idea: str = Field("", description="The core topic of the document that must be preserved.")
    previous_modifications: str = Field("", description="Summary of previous modifications to preserve.")
    context_before: str = Field("", description="Read-only context before the target content (DO NOT modify).")
    context_after: str = Field("", description="Read-only context after the target content (DO NOT modify).")

def apply_persuasion(strategy: str, target_content: str, core_idea: str = "", previous_modifications: str = "", context_before: str = "", context_after: str = "") -> str:
    """Enhances content to be more persuasive using specific strategies while preserving the core idea."""
    llm = get_llm_from_config('geo_agent/config.yaml')

    strategy_instruction = PERSUASION_STRATEGIES.get(strategy, PERSUASION_STRATEGIES["authoritative_tone"])

    # 1. 构建上下文部分
    context_section = ""
    if context_before or context_after:
        context_section = """
📖 SURROUNDING CONTEXT (READ-ONLY - for understanding flow, DO NOT include in output):
"""
        if context_before:
            context_section += f"""
--- CONTENT BEFORE (READ-ONLY) ---
{context_before}
--- END CONTENT BEFORE ---
"""
        if context_after:
            context_section += f"""
--- CONTENT AFTER (READ-ONLY) ---
{context_after}
--- END CONTENT AFTER ---
"""

    # 2. 构建历史修改记录部分
    history_section = ""
    if previous_modifications:
        history_section = f"""
⚠️ PREVIOUS MODIFICATIONS (MUST PRESERVE):
{previous_modifications}
"""

    prompt = ChatPromptTemplate.from_template("""
You are an expert persuasive content writer.

⚠️ CORE IDEA (MUST PRESERVE): {core_idea}

{history_section}
{context_section}

TASK: Apply the "{strategy}" persuasion strategy to enhance the content's persuasiveness.

STRATEGY GUIDE:
{strategy_instruction}

RULES:
1. Keep the document focused on the core idea - DO NOT drift to other topics
2. PRESERVE all previous modifications - DO NOT undo or contradict them
3. Preserve ALL original information - only improve how it's presented
4. Maintain the same format (Markdown/HTML) as the input
5. DO NOT fabricate data, fake citations, or false claims
6. DO NOT add content unrelated to the core subject
7. ONLY output the modified TARGET CONTENT section - DO NOT include the context sections

=== TARGET CONTENT (MODIFY THIS SECTION ONLY) ===
{target_content}
=== END TARGET CONTENT ===

OUTPUT FORMAT:
First output the complete enhanced TARGET CONTENT ONLY, then add a summary section:

[Your enhanced target content here - DO NOT include context sections...]

---MODIFICATION_SUMMARY---
- [Describe change 1 in one sentence]
- [Describe change 2 in one sentence]
- [Describe change 3 in one sentence]
""")

    response = llm.invoke(prompt.format(
        strategy=strategy,
        strategy_instruction=strategy_instruction,
        target_content=target_content,
        core_idea=core_idea or "Not specified",
        history_section=history_section,
        context_section=context_section
    ))
    return response.content

registry.register(
    apply_persuasion,
    PersuasionInput,
    name="persuasive_rewriting",
    description="Enhances content persuasiveness using specific strategies (authoritative_tone, counter_argument, emotional_hook, social_proof, scarcity_urgency, logical_structure) while preserving the core idea."
)
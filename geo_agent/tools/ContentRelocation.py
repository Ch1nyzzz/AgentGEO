"""
Content Relocation Tool
专门处理"关键信息被截断"的场景
通过重组文档结构，将隐藏的高价值内容移动到文档前部
"""
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from geo_agent.config import get_llm_from_config
from .registry import registry
from .utils import build_context_section, PRESERVATION_RULES


class ContentRelocationInput(BaseModel):
    target_content: str = Field(..., description="The current visible content (truncated view).")
    hidden_content_summary: str = Field(..., description="Summary of the relevant information found in the hidden/truncated part.")
    query: str = Field(..., description="The user query that triggered the optimization.")
    context_before: str = Field("", description="Read-only context before.")
    context_after: str = Field("", description="Read-only context after.")
    core_idea: str = Field("", description="The core topic of the document that must be preserved.")
    previous_modifications: str = Field("", description="Summary of previous modifications.")


def relocate_content(
    target_content: str, 
    hidden_content_summary: str,
    query: str,
    context_before: str = "", 
    context_after: str = "", 
    core_idea: str = "", 
    previous_modifications: str = ""
) -> str:
    """
    Restructures the document to surface hidden relevant content.

    Strategy:
    1. Analyze content style (formal/technical/casual/news/e-commerce)
    2. Dynamically choose appropriate header (Summary/TL;DR/Quick Answer/Highlights/etc.)
    3. Create a summary section at the TOP matching the page's tone
    4. Keep the original structure intact below
    """
    llm = get_llm_from_config('geo_agent/config.yaml')
    context_section = build_context_section(context_before, context_after)

    history_section = ""
    if previous_modifications:
        history_section = f"""
⚠️ PREVIOUS MODIFICATIONS (MUST PRESERVE):
{previous_modifications}
"""

    prompt = ChatPromptTemplate.from_template("""
You are a Content Restructuring Expert. The document has CRITICAL information buried deep (truncated from view).
Your job is to SURFACE this information by creating a prominent summary section at the TOP.

### Context
- **User Query**: "{query}"
- **Core Idea**: {core_idea}
- **Problem**: Important information was found in the truncated part of the document and is not visible to the search engine.

### Hidden Content Summary (from truncated section)
{hidden_summary}

{context_section}
{history_section}

{preservation_rules}

### STEP 1: Analyze Content Style
First, analyze the target content to determine its style:
- **Formal/Academic**: Research papers, official docs, business reports → Use "Summary" or "Key Findings"
- **Technical/Documentation**: API docs, tutorials, technical guides → Use "Quick Answer" or "Key Points"
- **Blog/Casual**: Personal blogs, forums, community posts → Use "TL;DR" or "The Short Version"
- **News/Editorial**: News articles, reviews, editorials → Use "Highlights" or "At a Glance"
- **E-commerce/Product**: Product pages, listings → Use "Quick Facts" or "Overview"

### STEP 2: Create Summary Section
1. **Choose the most appropriate header** based on the content style detected above.
2. **Synthesize** the hidden content summary into 2-4 bullet points that DIRECTLY answer the query.
3. **Use semantic HTML**: Wrap in `<div class="geo-summary-box">`, use `<h3>` for the header, `<ul><li>` for points.
4. **Match the tone**: The summary should feel native to the page - formal content gets formal language, casual content gets casual language.
5. **Preserve ALL original content** below this new section - DO NOT delete anything.

### Style Matching Rules
- Be concise but include specific facts, numbers, entities from the hidden summary
- Mirror the vocabulary and tone of the original content
- The new section should blend seamlessly with the existing page

=== TARGET CONTENT (Current Visible) ===
{target_content}
=== END TARGET CONTENT ===

OUTPUT FORMAT:
[New Summary Section with style-appropriate header]
[Original Content Preserved Below]

---MODIFICATION_SUMMARY---
- Detected content style: [style type]
- Added "[chosen header]" section at top with [specific facts surfaced]
""")

    response = llm.invoke(prompt.format(
        target_content=target_content,
        hidden_summary=hidden_content_summary,
        query=query,
        context_section=context_section,
        preservation_rules=PRESERVATION_RULES,
        core_idea=core_idea,
        history_section=history_section
    ))
    return response.content


# Register the tool
registry.register(
    relocate_content,
    ContentRelocationInput,
    name="content_relocation",
    description="Surfaces hidden/truncated relevant content by creating a style-matched summary section at the document top. Automatically detects content style (formal/technical/casual/news/e-commerce) and chooses appropriate header (Summary/TL;DR/Quick Answer/Highlights/etc.). Use when diagnosis indicates 'Content Truncated' and hidden content contains query-relevant information."
)

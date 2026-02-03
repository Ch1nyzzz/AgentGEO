from typing import List, Dict, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class TruncationAuditResult(BaseModel):
    has_hidden_relevant_content: bool = Field(description="True if the truncated (hidden) text contains information highly relevant to the query.")
    summary_of_hidden_info: str = Field(description="A concise summary of the useful information found in the hidden text. Empty if none found.")
    recommended_action: Optional[str] = Field(default=None, description="The recommended strategy (e.g., 'Move hidden content to top').")

def audit_content_truncation(
    llm,
    query: str,
    full_text: str,
    visible_chunks_text: str
) -> TruncationAuditResult:
    """
    Checks if the content that was truncated (not in visible_chunks) contains pertinent info for the query.

    Improved implementation:
    1. More accurate calculation of hidden_text
    2. Preprocess visible_chunks_text (remove CHUNK_ID markers)
    """

    # Preprocessing: Extract pure text from visible_chunks_text (remove >> [CHUNK_ID: X] markers)
    import re
    visible_clean = re.sub(r'>>\s*\[CHUNK_ID:\s*\d+\]\n?', '', visible_chunks_text).strip()

    # 1. Determine if audit is needed
    # If visible and full lengths are similar, no truncation occurred
    len_diff = len(full_text) - len(visible_clean)
    if len_diff < 200:  # Difference less than 200 characters, no audit needed
        return TruncationAuditResult(
            has_hidden_relevant_content=False, 
            summary_of_hidden_info="", 
            recommended_action=None
        )

    # 2. Extract hidden_text
    # Strategy: Find position of visible_clean in full_text, take the part after it
    # Since visible starts from the beginning (per calculate_chunks logic), we take the tail of full_text

    # Use first 300 characters of visible_clean as anchor to confirm alignment
    anchor_length = min(300, len(visible_clean))
    anchor = visible_clean[:anchor_length]

    # Simplification: If first 300 characters match, visible is a prefix of full
    # Directly take full_text[len(visible_clean):] as hidden
    if full_text[:anchor_length] == anchor or anchor in full_text[:anchor_length + 100]:
        # Roughly aligned, take tail
        # But visible_clean may be shorter than actual (due to newline handling etc.), use conservative estimate
        estimated_visible_end = int(len(visible_clean) * 0.9)  # Conservative estimate
        hidden_text = full_text[estimated_visible_end:]
    else:
        # Cannot determine alignment, use last 30%
        hidden_text = full_text[int(len(full_text) * 0.7):]
        
    if len(hidden_text) < 100:
        return TruncationAuditResult(has_hidden_relevant_content=False, summary_of_hidden_info="", recommended_action=None)

    # 3. Call LLM to verify relevance
    # Limit hidden_text length to save tokens
    hidden_text_preview = hidden_text[:12000]

    parser = PydanticOutputParser(pydantic_object=TruncationAuditResult)
    
    prompt = ChatPromptTemplate.from_template("""
    You are a Content Auditor checking for lost information due to truncation.
    
    User Query: "{query}"
    
    The following text is hidden/truncated from the main view because it was too far down the page.
    Check if this HIDDEN TEXT contains any specific answers, facts, or sections that are CRITICAL to the User Query.
    
    HIDDEN TEXT (Subset):
    {hidden_text}
    
    Task:
    1. Does this text contain relevant answers to the query?
    2. If yes, summarize the key information that should be surfaced.
    3. Suggest an action (e.g., "Move section about X to top").
    
    {format_instructions}
    """)
    
    final_prompt = prompt.format(
        query=query,
        hidden_text=hidden_text_preview,
        format_instructions=parser.get_format_instructions()
    )
    
    try:
        res = llm.invoke(final_prompt)
        return parser.parse(res.content)
    except Exception as e:
        print(f"Audit failed: {e}")
        return TruncationAuditResult(has_hidden_relevant_content=False, summary_of_hidden_info="", recommended_action=None)

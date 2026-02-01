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
    
    改进的实现：
    1. 更准确地计算 hidden_text
    2. 对 visible_chunks_text 做预处理（移除 CHUNK_ID 标记）
    """
    
    # 预处理：从 visible_chunks_text 中提取纯文本（移除 >> [CHUNK_ID: X] 标记）
    import re
    visible_clean = re.sub(r'>>\s*\[CHUNK_ID:\s*\d+\]\n?', '', visible_chunks_text).strip()
    
    # 1. 判断是否需要审计
    # 如果 visible 和 full 长度相近，说明没有截断
    len_diff = len(full_text) - len(visible_clean)
    if len_diff < 200:  # 差异小于 200 字符，认为无需审计
        return TruncationAuditResult(
            has_hidden_relevant_content=False, 
            summary_of_hidden_info="", 
            recommended_action=None
        )

    # 2. 提取 hidden_text
    # 策略：找到 visible_clean 在 full_text 中的位置，取后面的部分
    # 由于 visible 是从头开始的（按 calculate_chunks 逻辑），我们取 full_text 的尾部
    
    # 使用 visible_clean 的前 300 字符作为锚点确认对齐
    anchor_length = min(300, len(visible_clean))
    anchor = visible_clean[:anchor_length]
    
    # 简化：如果前 300 字符匹配，说明 visible 是 full 的前缀
    # 直接取 full_text[len(visible_clean):] 作为 hidden
    if full_text[:anchor_length] == anchor or anchor in full_text[:anchor_length + 100]:
        # 大致对齐，取尾部
        # 但 visible_clean 可能比实际更短（因为换行处理等），我们取一个保守估计
        estimated_visible_end = int(len(visible_clean) * 0.9)  # 保守估计
        hidden_text = full_text[estimated_visible_end:]
    else:
        # 无法确定对齐，使用尾部 30%
        hidden_text = full_text[int(len(full_text) * 0.7):]
        
    if len(hidden_text) < 100:
        return TruncationAuditResult(has_hidden_relevant_content=False, summary_of_hidden_info="", recommended_action=None)

    # 3. 调用 LLM 验证相关性
    # 限制 hidden_text 长度以节省 token
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

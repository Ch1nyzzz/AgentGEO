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

def inject_entities(missing_entities: str, target_content: str, context_before: str = "", context_after: str = "", core_idea: str = "", previous_modifications: str = "") -> str:
    """Weaves missing specific entities/facts naturally into the HTML content."""
    llm = get_llm_from_config('geo_agent/config.yaml')
    
    # 1. 预处理：确保 target_content 是有效的 HTML 片段
    # EntityInjection 需要上下文来决定插入位置，所以我们传递 HTML 给 LLM
    # 但我们使用 Processor 来确保最终输出的规范性
    processor = HTMLFragmentProcessor(target_content)
    # 获取纯文本供 LLM 参考（可选，但有时候 LLM看 HTML 结构更好定位）
    # 这里我们直接把 HTML 给 LLM，因为需要保留原有标签结构
    
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
    
    # 2. 解析 & 验证
    # modified_html, mod_summary = parse_tool_output(response.content)
    
    # 这里我们直接返回 LLM 修改后的 HTML，因为是“微创手术”，Python 很难帮它定位插入点。
    # 但我们可以用 Processor 简单清洗一下确保它是闭合的（可选）
    # processor_out = HTMLFragmentProcessor(modified_html)
    # final_html = processor_out.to_html()
    
    return response.content #f"{modified_html}\n\n---MODIFICATION_SUMMARY---\n{mod_summary}"

registry.register(inject_entities, EntityInjectionInput, name="entity_injection", description="Injects missing specific facts or entities into the HTML content naturally, using strong tags for emphasis.")
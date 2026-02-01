"""
Memory Module for GEO Optimization
记录修改历史，防止后续优化覆盖之前的改进
"""
from datetime import datetime
from typing import List, Tuple
from pydantic import BaseModel, Field


class ModificationRecord(BaseModel):
    """单次修改记录"""
    query: str = Field(..., description="为哪个 query 做的修改")
    tool_name: str = Field(..., description="使用的工具")
    reasoning: str = Field("", description="为什么做这个修改")
    key_changes: List[str] = Field(default_factory=list, description="关键改动摘要")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class OptimizationMemory:
    """优化记忆模块 - 管理修改历史"""

    def __init__(self, core_idea: str = ""):
        self.core_idea = core_idea
        self.modifications: List[ModificationRecord] = []

    def add_modification(self, record: ModificationRecord) -> None:
        """添加一条修改记录"""
        self.modifications.append(record)

    def get_history_summary(self) -> str:
        """生成历史修改摘要，用于注入到 prompt 中"""
        if not self.modifications:
            return "No previous modifications."

        summary_parts = []
        for i, mod in enumerate(self.modifications, 1):
            changes = "\n".join(f"    - {c}" for c in mod.key_changes)
            summary_parts.append(
                f"[Modification {i}] Query: {mod.query}\n"
                f"  Tool: {mod.tool_name}\n"
                f"  Reason: {mod.reasoning}\n"
                f"  Key Changes:\n{changes}"
            )
        return "\n\n".join(summary_parts)

    def get_preservation_rules(self) -> str:
        """生成保护之前修改的规则，注入到工具 prompt 中"""
        if not self.modifications:
            return ""

        all_changes = []
        for mod in self.modifications:
            all_changes.extend(mod.key_changes)

        if not all_changes:
            return ""

        rules = "⚠️ PREVIOUS MODIFICATIONS (MUST PRESERVE):\n"
        for change in all_changes:
            rules += f"- {change}\n"
        rules += "\nDO NOT remove, contradict, or undo any of these changes."
        return rules

    def get_recent_modifications(self, n: int = 3) -> List[ModificationRecord]:
        """获取最近 n 条修改记录"""
        return self.modifications[-n:] if self.modifications else []

    def to_dict(self) -> dict:
        """序列化为字典，用于日志保存"""
        return {
            "core_idea": self.core_idea,
            "modifications": [
                {
                    "query": m.query,
                    "tool_name": m.tool_name,
                    "reasoning": m.reasoning,
                    "key_changes": m.key_changes,
                    "timestamp": m.timestamp
                }
                for m in self.modifications
            ]
        }


# 工具输出分隔符
MODIFICATION_SUMMARY_SEPARATOR = "---MODIFICATION_SUMMARY---"


def _clean_html_content(content: str) -> str:
    """
    清洗 LLM 输出的 HTML 内容

    处理以下情况：
    - 移除代码块标记（```html ... ```）
    - 验证是否有 HTML 标签
    - 纯文本包装成 <p> 标签
    """
    # 移除代码块标记
    if content.startswith("```html"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    # 如果内容为空，返回空
    if not content:
        return ""

    # 验证是否有 HTML 标签，如果是纯文本则包装
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(content, 'html.parser')
    # 检查是否有任何标签
    if not soup.find():
        # 纯文本，包装成 <p>
        content = f"<p>{content}</p>"

    return content


def parse_tool_output(raw_output: str) -> Tuple[str, List[str]]:
    """
    解析工具输出，分离内容和改动摘要

    Args:
        raw_output: 工具的原始输出，格式为:
            [修改后的完整内容]

            ---MODIFICATION_SUMMARY---
            - 改动1
            - 改动2

    Returns:
        Tuple[str, List[str]]: (纯净内容, 关键改动列表)
    """
    if MODIFICATION_SUMMARY_SEPARATOR in raw_output:
        parts = raw_output.split(MODIFICATION_SUMMARY_SEPARATOR, 1)
        content = parts[0].strip()
        summary_section = parts[1].strip()
    else:
        content = raw_output.strip()
        summary_section = ""

    # 清洗代码块标记（LLM 可能返回 ```html ... ```）
    content = _clean_html_content(content)

    # 解析改动列表
    key_changes = []
    if summary_section:
        for line in summary_section.split("\n"):
            line = line.strip()
            if line:
                # 移除常见的列表标记
                clean_line = line.lstrip("-•*").strip()
                if clean_line:
                    key_changes.append(clean_line)

    return content, key_changes

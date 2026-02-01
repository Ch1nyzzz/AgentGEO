"""
Paragraph Module for GEO Optimization
使用 RecursiveCharacterTextSplitter 按语义边界分段
"""
from typing import List, Optional, Tuple
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Paragraph:
    """段落"""
    index: int
    content: str
    token_estimate: int
    start_pos: int = 0  # 在原文中的起始位置
    end_pos: int = 0    # 在原文中的结束位置


class ParagraphManager:
    """段落管理器 - 使用 tiktoken 按语义边界分段"""

    def __init__(self, chunk_size: int = 2500, chunk_overlap: int = 0):
        """
        Args:
            chunk_size: 目标段落大小 (tokens)
            chunk_overlap: 段落重叠大小 (tokens)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", ", ", " ", ""],
        )

    def estimate_tokens(self, text: str) -> int:
        """估算 token 数（使用 tiktoken）"""
        try:
            import tiktoken
            enc = tiktoken.encoding_for_model("gpt-4")
            return len(enc.encode(text))
        except Exception:
            return int(len(text) / 4.0)

    def split(self, text: str) -> List[Paragraph]:
        """按语义边界分段（优先段落 > 行 > 句子 > 单词）"""
        self._original_text = text  # 保存原文用于 merge_back
        chunks = self._splitter.split_text(text)
        paragraphs = []

        # 记录每个 chunk 在原文中的位置（处理 overlap）
        search_start = 0
        for i, chunk in enumerate(chunks):
            # 在原文中查找 chunk 的位置
            start_pos = text.find(chunk, search_start)
            if start_pos == -1:
                # 如果找不到完整匹配（可能因为 overlap），从头找
                start_pos = text.find(chunk[:100], search_start)
                if start_pos == -1:
                    start_pos = search_start
            end_pos = start_pos + len(chunk)

            paragraphs.append(Paragraph(
                index=i,
                content=chunk,
                token_estimate=self.estimate_tokens(chunk),
                start_pos=start_pos,
                end_pos=end_pos,
            ))

            # 下一个 chunk 从当前结束位置减去 overlap 开始搜索
            search_start = max(start_pos + 1, end_pos - self.chunk_overlap * 4)

        return paragraphs

    def get_with_context(
        self,
        paragraphs: List[Paragraph],
        target_index: int
    ) -> Tuple[Optional[str], str, Optional[str]]:
        """获取目标段落及上下文"""
        if not paragraphs or target_index < 0 or target_index >= len(paragraphs):
            raise ValueError(f"Invalid target_index: {target_index}")

        context_before = paragraphs[target_index - 1].content if target_index > 0 else None
        target_content = paragraphs[target_index].content
        context_after = paragraphs[target_index + 1].content if target_index < len(paragraphs) - 1 else None

        return context_before, target_content, context_after

    def merge_back(
        self,
        paragraphs: List[Paragraph],
        target_index: int,
        modified_content: str
    ) -> str:
        """将修改后的段落合并回原文（使用位置信息精确替换）"""
        if not hasattr(self, '_original_text'):
            # 兼容：如果没有原文，直接拼接
            result = []
            for p in paragraphs:
                if p.index == target_index:
                    result.append(modified_content)
                else:
                    result.append(p.content)
            return ''.join(result)

        # 使用位置信息在原文中替换
        target = paragraphs[target_index]
        original = self._original_text
        return original[:target.start_pos] + modified_content + original[target.end_pos:]

    def get_summary(self, paragraphs: List[Paragraph]) -> str:
        """生成段落摘要，用于 LLM 选择"""
        parts = []
        for p in paragraphs:
            preview = p.content[:100].replace('\n', ' ').strip()
            if len(p.content) > 100:
                preview += "..."
            parts.append(f"[Paragraph {p.index}] (~{p.token_estimate} tokens): {preview}")

        return "\n".join(parts)

    def format_for_prompt(self, paragraphs: List[Paragraph]) -> str:
        """生成带标注的完整段落列表，用于 LLM prompt"""
        parts = []
        for p in paragraphs:
            parts.append(f"=== Paragraph {p.index} ===\n{p.content}")
        return "\n\n".join(parts)


# 兼容旧接口
ContentChunk = Paragraph

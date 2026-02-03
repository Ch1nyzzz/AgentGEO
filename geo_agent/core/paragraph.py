"""
Paragraph Module for GEO Optimization
Uses RecursiveCharacterTextSplitter to split by semantic boundaries
"""
from typing import List, Optional, Tuple
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Paragraph:
    """Paragraph"""
    index: int
    content: str
    token_estimate: int
    start_pos: int = 0  # Start position in original text
    end_pos: int = 0    # End position in original text


class ParagraphManager:
    """Paragraph Manager - Uses tiktoken to split by semantic boundaries"""

    def __init__(self, chunk_size: int = 2500, chunk_overlap: int = 0):
        """
        Args:
            chunk_size: Target paragraph size (tokens)
            chunk_overlap: Paragraph overlap size (tokens)
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
        """Estimate token count (using tiktoken)"""
        try:
            import tiktoken
            enc = tiktoken.encoding_for_model("gpt-4")
            return len(enc.encode(text))
        except Exception:
            return int(len(text) / 4.0)

    def split(self, text: str) -> List[Paragraph]:
        """Split by semantic boundaries (priority: paragraph > line > sentence > word)"""
        self._original_text = text  # Save original text for merge_back
        chunks = self._splitter.split_text(text)
        paragraphs = []

        # Record position of each chunk in original text (handle overlap)
        search_start = 0
        for i, chunk in enumerate(chunks):
            # Find chunk position in original text
            start_pos = text.find(chunk, search_start)
            if start_pos == -1:
                # If no complete match found (possibly due to overlap), search from beginning
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

            # Next chunk search starts from current end position minus overlap
            search_start = max(start_pos + 1, end_pos - self.chunk_overlap * 4)

        return paragraphs

    def get_with_context(
        self,
        paragraphs: List[Paragraph],
        target_index: int
    ) -> Tuple[Optional[str], str, Optional[str]]:
        """Get target paragraph and context"""
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
        """Merge modified paragraph back into original text (using position info for precise replacement)"""
        if not hasattr(self, '_original_text'):
            # Compatibility: if no original text, concatenate directly
            result = []
            for p in paragraphs:
                if p.index == target_index:
                    result.append(modified_content)
                else:
                    result.append(p.content)
            return ''.join(result)

        # Use position info to replace in original text
        target = paragraphs[target_index]
        original = self._original_text
        return original[:target.start_pos] + modified_content + original[target.end_pos:]

    def get_summary(self, paragraphs: List[Paragraph]) -> str:
        """Generate paragraph summary for LLM selection"""
        parts = []
        for p in paragraphs:
            preview = p.content[:100].replace('\n', ' ').strip()
            if len(p.content) > 100:
                preview += "..."
            parts.append(f"[Paragraph {p.index}] (~{p.token_estimate} tokens): {preview}")

        return "\n".join(parts)

    def format_for_prompt(self, paragraphs: List[Paragraph]) -> str:
        """Generate annotated complete paragraph list for LLM prompt"""
        parts = []
        for p in paragraphs:
            parts.append(f"=== Paragraph {p.index} ===\n{p.content}")
        return "\n\n".join(parts)


# Compatibility with old interface
ContentChunk = Paragraph

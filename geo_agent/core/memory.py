"""
Memory Module for GEO Optimization
Records modification history to prevent subsequent optimizations from overwriting previous improvements
"""
from datetime import datetime
from typing import List, Tuple
from pydantic import BaseModel, Field


class ModificationRecord(BaseModel):
    """Single modification record"""
    query: str = Field(..., description="The query this modification was made for")
    tool_name: str = Field(..., description="The tool used")
    reasoning: str = Field("", description="Why this modification was made")
    key_changes: List[str] = Field(default_factory=list, description="Summary of key changes")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class OptimizationMemory:
    """Optimization memory module - manages modification history"""

    def __init__(self, core_idea: str = ""):
        self.core_idea = core_idea
        self.modifications: List[ModificationRecord] = []

    def add_modification(self, record: ModificationRecord) -> None:
        """Add a modification record"""
        self.modifications.append(record)

    def get_history_summary(self) -> str:
        """Generate historical modification summary for injection into prompts"""
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
        """Generate rules for preserving previous modifications, injected into tool prompts"""
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
        """Get the most recent n modification records"""
        return self.modifications[-n:] if self.modifications else []

    def to_dict(self) -> dict:
        """Serialize to dictionary for log storage"""
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


# Tool output separator
MODIFICATION_SUMMARY_SEPARATOR = "---MODIFICATION_SUMMARY---"


def _clean_html_content(content: str) -> str:
    """
    Clean HTML content from LLM output

    Handles the following cases:
    - Remove code block markers (```html ... ```)
    - Validate presence of HTML tags
    - Wrap plain text in <p> tags
    """
    # Remove code block markers
    if content.startswith("```html"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    # If content is empty, return empty
    if not content:
        return ""

    # Validate HTML tags, wrap in tags if plain text
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(content, 'html.parser')
    # Check if there are any tags
    if not soup.find():
        # Plain text, wrap in <p>
        content = f"<p>{content}</p>"

    return content


def parse_tool_output(raw_output: str) -> Tuple[str, List[str]]:
    """
    Parse tool output, separate content and modification summary

    Args:
        raw_output: Raw output from tool, format:
            [Modified complete content]

            ---MODIFICATION_SUMMARY---
            - Change 1
            - Change 2

    Returns:
        Tuple[str, List[str]]: (Clean content, list of key changes)
    """
    if MODIFICATION_SUMMARY_SEPARATOR in raw_output:
        parts = raw_output.split(MODIFICATION_SUMMARY_SEPARATOR, 1)
        content = parts[0].strip()
        summary_section = parts[1].strip()
    else:
        content = raw_output.strip()
        summary_section = ""

    # Clean code block markers (LLM may return ```html ... ```)
    content = _clean_html_content(content)

    # Parse list of changes
    key_changes = []
    if summary_section:
        for line in summary_section.split("\n"):
            line = line.strip()
            if line:
                # Remove common list markers
                clean_line = line.lstrip("-•*").strip()
                if clean_line:
                    key_changes.append(clean_line)

    return content, key_changes

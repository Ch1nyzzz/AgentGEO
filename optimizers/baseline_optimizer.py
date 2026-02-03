"""
GEO-Bench Baseline optimizer wrapper

Supports 9 baseline optimization methods
Reference: GEO: Generative Engine Optimization (2311.09735v3) Section 2.2.2
"""
import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Add project path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from geo_agent.config import (
    load_config,
    GeminiChatLLM,
    OpenAIChatLLM,
    AnthropicChatLLM,
)
from geo_agent.utils.structural_parser import StructuralHtmlParser

logger = logging.getLogger(__name__)

# 9 optimization methods supported by GEO-Bench
BASELINE_METHODS = [
    "authoritative",
    "cite_sources",
    "statistics_addition",
    "keyword_stuffing",
    "quotation_addition",
    "easy_to_understand",
    "fluency_optimization",
    "unique_words",
    "technical_terms",
]

# Prompt templates for 9 GEO methods
PROMPTS = {
    "authoritative": (
        "You are given a website source text. "
        "Rewrite the text to make it more persuasive and authoritative. "
        "Use a confident, expert tone while maintaining the same core information. "
        "Only output the modified text, do not include any explanations.\n\n"
        "Source text:\n{content}"
    ),
    "statistics_addition": (
        "You are given a website source text. "
        "Modify the text to include relevant quantitative statistics, data points, "
        "and numerical evidence wherever qualitative claims are made. "
        "Add specific numbers, percentages, and data to support claims. "
        "Only output the modified text, do not include any explanations.\n\n"
        "Source text:\n{content}"
    ),
    "keyword_stuffing": (
        "You are given a website source text. "
        "Identify the main topics and themes in this text, then rewrite it to "
        "naturally incorporate more related keywords and key phrases throughout the content. "
        "Only output the modified text, do not include any explanations.\n\n"
        "Source text:\n{content}"
    ),
    "cite_sources": (
        "You are given a website source text. "
        "Add relevant citations and references to credible, authoritative sources "
        "throughout the text. Include academic papers, official reports, or well-known "
        "experts where appropriate. "
        "Only output the modified text, do not include any explanations.\n\n"
        "Source text:\n{content}"
    ),
    "quotation_addition": (
        "You are given a website source text. "
        "Add relevant direct quotations from credible experts, researchers, or "
        "authoritative figures to support the claims made in the text. "
        "Only output the modified text, do not include any explanations.\n\n"
        "Source text:\n{content}"
    ),
    "easy_to_understand": (
        "You are given a website source text. "
        "Rewrite the text to make it simpler and easier to understand. "
        "Use plain language, shorter sentences, and avoid jargon while preserving "
        "all key information. "
        "Only output the modified text, do not include any explanations.\n\n"
        "Source text:\n{content}"
    ),
    "fluency_optimization": (
        "You are given a website source text. "
        "Improve the fluency, readability, and flow of the text. "
        "Fix any grammatical issues, improve sentence structure, and make the writing "
        "smoother and more polished. "
        "Only output the modified text, do not include any explanations.\n\n"
        "Source text:\n{content}"
    ),
    "unique_words": (
        "You are given a website source text. "
        "Rewrite the text to include more unique, distinctive, and varied vocabulary. "
        "Replace common words with more specific and unique alternatives where appropriate. "
        "Only output the modified text, do not include any explanations.\n\n"
        "Source text:\n{content}"
    ),
    "technical_terms": (
        "You are given a website source text. "
        "Rewrite the text to incorporate more relevant technical terminology and "
        "domain-specific language where appropriate, while keeping the content comprehensible. "
        "Only output the modified text, do not include any explanations.\n\n"
        "Source text:\n{content}"
    ),
}


@dataclass
class BaselineResult:
    """Baseline optimization result"""
    optimized_text: str
    optimized_html: str
    original_text: str
    method_name: str


class BaselineOptimizers:
    """
    GEO-Bench Baseline optimizer

    Supports 9 baseline optimization methods:
    - authoritative: Make text more authoritative
    - cite_sources: Add citation sources
    - statistics_addition: Add statistical data
    - keyword_stuffing: Keyword enhancement
    - quotation_addition: Add quotations
    - easy_to_understand: Simplify for understanding
    - fluency_optimization: Fluency optimization
    - unique_words: Unique vocabulary
    - technical_terms: Technical terminology
    """

    def __init__(
        self,
        method_name: str,
        provider: str = "openai",
        model: str = "gpt-4.1-mini",
        config_path: str = "geo_agent/config.yaml",
        temperature: float = 0.7,
        **kwargs
    ):
        """
        Initialize Baseline optimizer

        Args:
            method_name: Optimization method name
            provider: LLM provider (openai, anthropic, gemini)
            model: Model name
            config_path: Configuration file path
            temperature: Generation temperature
        """
        if method_name not in BASELINE_METHODS:
            raise ValueError(
                f"Invalid method: {method_name}. "
                f"Must be one of {BASELINE_METHODS}"
            )
        self.method_name = method_name
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self._parser = StructuralHtmlParser()

        # Load configuration
        config_file = Path(config_path)
        if not config_file.is_absolute():
            config_file = (REPO_ROOT / config_file).resolve()
        self._config = load_config(str(config_file))

        # Initialize LLM
        self._llm = self._create_llm()
        logger.info(f"BaselineOptimizers initialized with method: {method_name}")

    def _create_llm(self):
        """Create LLM client"""
        if self.provider == "openai" or "gpt" in self.model.lower():
            return OpenAIChatLLM(model=self.model, temperature=self.temperature)
        elif self.provider == "anthropic" or "claude" in self.model.lower():
            return AnthropicChatLLM(model=self.model, temperature=self.temperature)
        elif self.provider == "gemini" or "gemini" in self.model.lower():
            return GeminiChatLLM(model=self.model, temperature=self.temperature)
        else:
            # Default to OpenAI
            return OpenAIChatLLM(model="gpt-4.1-mini", temperature=self.temperature)

    def _build_prompt(self, text: str) -> str:
        """Build optimization prompt"""
        return PROMPTS[self.method_name].format(content=text)

    def extract_clean_text(self, raw_html: str) -> str:
        """Extract plain text from raw_html"""
        try:
            structure = self._parser.parse(raw_html)
            return structure.get_clean_text()
        except Exception as e:
            logger.error(f"Failed to extract clean text: {e}")
            return raw_html

    async def optimize_async(
        self,
        raw_html: str,
        train_queries: List[str] = None,
        url: str = ""
    ) -> BaselineResult:
        """Async document optimization"""
        # Extract original text
        original_text = self.extract_clean_text(raw_html)

        if not original_text.strip():
            logger.warning("Extracted clean_text is empty")
            return BaselineResult(
                optimized_text=raw_html,
                optimized_html=raw_html,
                original_text=raw_html,
                method_name=self.method_name
            )

        logger.info(f"Optimizing with {self.method_name} (text length: {len(original_text)})")

        # Build prompt and call LLM
        prompt = self._build_prompt(original_text)

        try:
            response = await self._llm.ainvoke(prompt)
            optimized_text = response.content.strip()
            logger.info(f"Baseline {self.method_name} completed (output length: {len(optimized_text)})")

            return BaselineResult(
                optimized_text=optimized_text,
                optimized_html=optimized_text,
                original_text=original_text,
                method_name=self.method_name
            )
        except Exception as e:
            logger.error(f"Baseline {self.method_name} failed: {e}")
            return BaselineResult(
                optimized_text=original_text,
                optimized_html=raw_html,
                original_text=original_text,
                method_name=self.method_name
            )

    def optimize(
        self,
        raw_html: str,
        train_queries: List[str] = None,
        url: str = ""
    ) -> BaselineResult:
        """Sync optimization interface"""
        return asyncio.run(self.optimize_async(raw_html, train_queries, url))

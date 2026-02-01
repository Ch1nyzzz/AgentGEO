"""
GEO 论文中 9 种 baseline 优化方法的统一优化器

参考: GEO: Generative Engine Optimization (2311.09735v3) Section 2.2.2
"""
import asyncio
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class SimpleLLMClient:
    """简单的 LLM 客户端，支持 OpenAI 和 Anthropic"""

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key

        if self.provider == "openai":
            self.model = model or "gpt-4.1-mini"
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        elif self.provider in ("anthropic", "claude"):
            self.provider = "anthropic"
            self.model = model or "claude-haiku-4-5-20251001"
            self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 4096) -> str:
        """同步生成"""
        if self.provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        elif self.provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.content[0].text

    async def async_generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 4096) -> str:
        """异步生成"""
        if self.provider == "openai":
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=self.api_key)
            response = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        elif self.provider == "anthropic":
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=self.api_key)
            response = await client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.content[0].text

# 9 种 GEO 方法的 Prompt 模板
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


class BaselineGEOOptimizer:
    """GEO 论文中 9 种 baseline 方法的统一优化器"""

    METHODS = list(PROMPTS.keys())

    def __init__(
        self,
        llm_provider: str = "openai",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        method_name: str = "authoritative",
        methods: Optional[List[str]] = None,  # 支持多方法模式
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        if method_name not in PROMPTS:
            raise ValueError(
                f"Unknown method: {method_name}. "
                f"Available: {self.METHODS}"
            )
        self.method_name = method_name
        self.methods = methods or [method_name]  # 支持多方法列表
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm_client = SimpleLLMClient(
            provider=llm_provider,
            api_key=api_key,
            model=model,
        )

    def _build_prompt(self, text: str) -> str:
        return PROMPTS[self.method_name].format(content=text)

    def optimize(self, text: str) -> str:
        """同步优化文本"""
        prompt = self._build_prompt(text)
        return self.llm_client.generate(
            prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    async def optimize_async(self, text: str) -> str:
        """异步优化文本（单方法）"""
        prompt = self._build_prompt(text)
        return await self.llm_client.async_generate(
            prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def optimize_all_methods(self, text: str) -> Dict[str, str]:
        """使用所有指定方法优化文本（同步）"""
        results = {}
        for method in self.methods:
            prompt = PROMPTS[method].format(content=text)
            results[method] = self.llm_client.generate(
                prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        return results

    async def optimize_all_methods_async(self, text: str) -> Dict[str, str]:
        """使用所有指定方法优化文本（异步并行）"""
        async def optimize_single(method: str) -> tuple:
            prompt = PROMPTS[method].format(content=text)
            result = await self.llm_client.async_generate(
                prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return (method, result)

        tasks = [optimize_single(m) for m in self.methods]
        results = await asyncio.gather(*tasks)
        return dict(results)

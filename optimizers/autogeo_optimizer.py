"""
AutoGEO 优化器封装

基于 AutoGEO 论文的规则重写方法
"""
import asyncio
import glob
import json
import logging
import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

# 添加项目路径
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


class Dataset(str, Enum):
    """数据集类型"""
    RESEARCHY_GEO = "Researchy-GEO"
    GEO_BENCH = "GEO-Bench"
    ECOMMERCE = "E-commerce"


class LLMName(str, Enum):
    """LLM 类型"""
    GEMINI = "gemini"
    GPT = "gpt"
    CLAUDE = "claude"


@dataclass
class AutoGEOResult:
    """AutoGEO 优化结果"""
    optimized_text: str
    optimized_html: str
    original_text: str
    rules_applied: List[str]


# ============== 规则定义 ==============

# Default rules for GEO-Bench (research queries) gemini
AUTOGEO_GEO_BENCH_GEMINI_RULES = [
    "Ensure all information is factually accurate and verifiable, citing credible sources.",
    "Ensure information is current and up-to-date, especially for time-sensitive topics.",
    "Ensure the document is self-contained and comprehensive, providing all necessary context and sub-topic information.",
    "Explain the underlying mechanisms and principles (the 'why' and 'how'), not just surface-level facts.",
    "Maintain a singular focus on the core topic, excluding tangential information, promotional content, and document 'noise' (e.g., navigation, ads).",
    "Organize content with a clear, logical hierarchy, using elements like headings, lists, and tables.",
    "Present a balanced and objective view on debatable topics, including multiple significant perspectives.",
    "Provide specific, actionable guidance, such as step-by-step instructions, for procedural topics.",
    "State the primary conclusion directly at the beginning of the document.",
    "Use clear and unambiguous language, defining technical terms, acronyms, and jargon upon first use.",
    "Use specific, concrete details and examples instead of abstract generalizations.",
    "Write concisely, eliminating verbose language, redundancy, and filler content."
]

# Default rules for GEO-Bench (research queries) gpt
AUTOGEO_GEO_BENCH_GPT_RULES = [
    "Address the topic comprehensively, covering all essential sub-topics and necessary context.",
    "Define essential terms, acronyms, and jargon upon their first use.",
    "Ensure all factual information is accurate, verifiable, and internally consistent.",
    "Ensure content is free from illegal, unethical, or harmful information.",
    "Ensure each document is self-contained, providing all necessary information on the topic without requiring external links.",
    "Explain the 'why' and 'how' behind facts, clarifying underlying principles and mechanisms.",
    "Explicitly differentiate between similar or easily confused concepts.",
    "For complex or debatable subjects, present multiple significant viewpoints in a balanced way.",
    "For procedural content, provide clear, numbered, step-by-step instructions.",
    "For time-sensitive topics, ensure information is current and clearly display its publication or last-updated date.",
    "Maintain a neutral, objective tone, clearly distinguishing facts from opinions.",
    "Maintain a singular focus on the core topic, excluding tangential or promotional content.",
    "Organize content with a clear, logical hierarchy using headings, lists, and tables.",
    "State the primary conclusion at the beginning of the document.",
    "Structure content into atomic units, where each paragraph or section addresses a single idea.",
    "Use clear, simple, and unambiguous language.",
    "Use concrete examples, analogies, or case studies to illustrate complex concepts.",
    "Use specific, concrete details like names, dates, and statistics instead of generalizations.",
    "Write concisely, eliminating repetition, filler words, and verbose phrasing."
]

# Default rules for GEO-Bench (research queries) claude
AUTOGEO_GEO_BENCH_CLAUDE_RULES = [
    "Cite authoritative sources to support claims and establish credibility.",
    "Cover the topic comprehensively, providing depth by explaining the underlying 'why' and 'how'.",
    "Ensure all information is factually accurate, verifiable, and internally consistent.",
    "Ensure each document is self-contained and can be understood without external context.",
    "Focus on a single topic, writing concisely and eliminating irrelevant or repetitive content.",
    "For task-oriented topics, provide actionable guidance like step-by-step instructions.",
    "Indicate the timeliness of information with clear publication or revision dates.",
    "Maintain a neutral, objective tone, prioritizing facts over opinions or promotional language.",
    "Present multiple perspectives and counterarguments for complex or debatable topics.",
    "Provide specific details, such as names, dates, statistics, and concrete examples, to support claims and illustrate concepts.",
    "Segment content into discrete units, where each paragraph or list item addresses a single idea.",
    "State the key conclusion at the beginning of the document.",
    "Use clear structural elements like headings, lists, and tables to organize content logically.",
    "Use clear, unambiguous language, and define technical terms or acronyms on their first use."
]

# Researchy-GEO rules
AUTOGEO_RESEARCHY_GEO_GEMINI_RULES = [
    "Attribute all factual claims to credible, authoritative sources with clear citations.",
    "Cover the topic comprehensively, addressing all key aspects and sub-topics.",
    "Ensure information is factually accurate and verifiable.",
    "Focus exclusively on the topic, eliminating irrelevant information, navigational links, and advertisements.",
    "Maintain a neutral, objective tone, avoiding promotional language, personal opinions, and bias.",
    "Maintain high-quality writing, free from grammatical errors, typos, and formatting issues.",
    "Present a balanced perspective on complex topics, acknowledging multiple significant viewpoints or counter-arguments.",
    "Present information as a self-contained unit, not requiring external links for core understanding.",
    "Provide clear, specific, and actionable steps.",
    "Provide explanatory depth by clarifying underlying causes, mechanisms, and context ('how' and 'why').",
    "State the key conclusion at the beginning of the document.",
    "Structure content logically with clear headings, lists, and paragraphs to ensure a cohesive flow.",
    "Substantiate claims with specific, concrete details like data, statistics, or named examples.",
    "Use clear and concise language, avoiding jargon, ambiguity, and verbosity.",
    "Use current information, reflecting the latest state of knowledge."
]

# E-commerce rules
AUTOGEO_ECOMMERCE_GEMINI_RULES = [
    "Ensure all information is factually accurate, verifiable, and current for the topic.",
    "Establish credibility by citing authoritative sources, providing evidence, or demonstrating clear expertise.",
    "Justify recommendations and claims with clear reasoning, context, or comparative analysis like pros and cons.",
    "Organize content with a clear, logical structure using elements like headings, lists, and tables to facilitate scanning and parsing.",
    "Present information objectively, avoiding promotional bias and including balanced perspectives where applicable.",
    "Provide actionable information, such as step-by-step instructions or clear recommendations.",
    "Provide specific, verifiable details such as names, model numbers, technical specifications, and quantifiable data.",
    "Structure content into modular, self-contained units, such as distinct paragraphs or list items for each concept.",
    "Use clear, simple, and unambiguous language, defining any necessary technical terms or jargon.",
    "Write concisely, eliminating verbose language, filler content, and unnecessary repetition."
]

# 规则映射表
DATASET_RULES_MAP = {
    (Dataset.GEO_BENCH, LLMName.GEMINI): AUTOGEO_GEO_BENCH_GEMINI_RULES,
    (Dataset.GEO_BENCH, LLMName.GPT): AUTOGEO_GEO_BENCH_GPT_RULES,
    (Dataset.GEO_BENCH, LLMName.CLAUDE): AUTOGEO_GEO_BENCH_CLAUDE_RULES,
    (Dataset.RESEARCHY_GEO, LLMName.GEMINI): AUTOGEO_RESEARCHY_GEO_GEMINI_RULES,
    (Dataset.RESEARCHY_GEO, LLMName.GPT): AUTOGEO_GEO_BENCH_GPT_RULES,
    (Dataset.RESEARCHY_GEO, LLMName.CLAUDE): AUTOGEO_GEO_BENCH_CLAUDE_RULES,
    (Dataset.ECOMMERCE, LLMName.GEMINI): AUTOGEO_ECOMMERCE_GEMINI_RULES,
    (Dataset.ECOMMERCE, LLMName.GPT): AUTOGEO_GEO_BENCH_GPT_RULES,
    (Dataset.ECOMMERCE, LLMName.CLAUDE): AUTOGEO_GEO_BENCH_CLAUDE_RULES,
}


def _load_rules_from_file(
    dataset: str,
    engine_llm: str,
    rule_path: Optional[str] = None
) -> Tuple[Optional[List[str]], Optional[str]]:
    """从文件加载规则"""
    if rule_path:
        try:
            with open(rule_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and 'filtered_rules' in data:
                    return data['filtered_rules'], rule_path
                if isinstance(data, list):
                    return data, rule_path
        except Exception as exc:
            logger.warning(f"Failed to load rules from {rule_path}: {exc}")

    # 尝试自动发现规则文件
    possible_paths = [
        f"data/{dataset}/rule_sets/{engine_llm}*/merged_rules.json",
        str(REPO_ROOT / f"data/{dataset}/rule_sets/{engine_llm}*/merged_rules.json"),
    ]

    for pattern in possible_paths:
        matches = glob.glob(pattern)
        if matches:
            latest_file = max(matches, key=os.path.getmtime)
            try:
                with open(latest_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and 'filtered_rules' in data:
                        return data['filtered_rules'], latest_file
                    elif isinstance(data, list):
                        return data, latest_file
            except Exception as e:
                logger.warning(f"Could not load rules from {latest_file}: {e}")

    return None, None


def _get_default_rules(dataset: str, engine_llm: str) -> List[str]:
    """获取默认规则"""
    try:
        dataset_enum = Dataset(dataset)
    except ValueError:
        dataset_enum = Dataset.GEO_BENCH

    # 从 engine_llm 提取 LLM 类型
    engine_llm_lower = engine_llm.lower()
    llm_enum = LLMName.GEMINI  # 默认

    for llm_name in LLMName:
        if llm_name.value in engine_llm_lower:
            llm_enum = llm_name
            break

    return DATASET_RULES_MAP.get(
        (dataset_enum, llm_enum),
        AUTOGEO_GEO_BENCH_GEMINI_RULES
    )


class AutoGEOOptimizer:
    """
    AutoGEO 优化器

    基于规则的文档重写方法，使用 LLM 根据预定义规则优化文档
    """

    def __init__(
        self,
        dataset_name: str = "GEO-Bench",
        engine_llm: str = "gemini",
        rule_path: Optional[str] = None,
        config_path: str = "geo_agent/config.yaml",
        **kwargs
    ):
        """
        初始化 AutoGEO 优化器

        Args:
            dataset_name: 数据集名称 (GEO-Bench, Researchy-GEO, E-commerce)
            engine_llm: LLM 类型 (gemini, gpt, claude)
            rule_path: 可选的自定义规则路径
            config_path: 配置文件路径
        """
        self.dataset_name = dataset_name
        self.engine_llm = engine_llm
        self.rule_path = rule_path
        self.config_path = config_path
        self._parser = StructuralHtmlParser()

        # 加载配置
        config_file = Path(config_path)
        if not config_file.is_absolute():
            config_file = (REPO_ROOT / config_file).resolve()
        self._config = load_config(str(config_file))

        # 初始化 LLM
        self._llm = self._create_llm()

        # 加载规则
        self.rules, self._rule_file = _load_rules_from_file(
            dataset_name, engine_llm, rule_path
        )
        if not self.rules:
            self.rules = _get_default_rules(dataset_name, engine_llm)
            logger.info(f"Using default rules for {dataset_name}/{engine_llm}")
        else:
            logger.info(f"Loaded rules from {self._rule_file}")

    def _create_llm(self):
        """创建 LLM 客户端"""
        engine_llm_lower = self.engine_llm.lower()

        # 从配置文件获取默认参数
        llm_config = self._config.get('llm', {})
        temperature = llm_config.get('temperature', 0.7)

        if "gpt" in engine_llm_lower or "openai" in engine_llm_lower:
            return OpenAIChatLLM(model="gpt-4.1-mini", temperature=temperature)
        elif "claude" in engine_llm_lower or "anthropic" in engine_llm_lower:
            return AnthropicChatLLM(model="claude-haiku-4-5-20251001", temperature=temperature)
        else:  # gemini
            return GeminiChatLLM(model="gemini-2.5-flash", temperature=temperature)

    def _build_prompt(self, document: str) -> str:
        """构建重写提示"""
        rules_string = "- " + "\n- ".join(self.rules)

        return f"""Here is the source:
{document}

You are given a website document as a source. This source, along with other sources, will be used by a language model (LLM) to generate answers to user questions, with each line in the generated answer being cited with its original source. Your task, as the owner of the source, is to **rewrite your document in a way that maximizes its visibility and impact in the LLM's final answer, ensuring your source is more likely to be quoted and cited**.

You can regenerate the provided source so that it strictly adheres to the "Quality Guidelines", and you can also apply any other methods or techniques, as long as they help your rewritten source text rank higher in terms of relevance, authority, and impact in the LLM's generated answers.

## Quality Guidelines to Follow:

{rules_string}

Output only the rewritten document text, without any explanations or preamble."""

    def extract_clean_text(self, raw_html: str) -> str:
        """从 raw_html 提取纯文本"""
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
    ) -> AutoGEOResult:
        """异步优化文档"""
        # 提取原始文本
        original_text = self.extract_clean_text(raw_html)

        if not original_text.strip():
            logger.warning("Extracted clean_text is empty")
            return AutoGEOResult(
                optimized_text=raw_html,
                optimized_html=raw_html,
                original_text=raw_html,
                rules_applied=[]
            )

        logger.info(f"Optimizing document with AutoGEO (text length: {len(original_text)})")

        # 构建提示并调用 LLM
        prompt = self._build_prompt(original_text)

        try:
            response = await self._llm.ainvoke(prompt)
            optimized_text = response.content.strip()
            logger.info(f"AutoGEO optimization completed (output length: {len(optimized_text)})")

            return AutoGEOResult(
                optimized_text=optimized_text,
                optimized_html=optimized_text,
                original_text=original_text,
                rules_applied=self.rules
            )
        except Exception as e:
            logger.error(f"AutoGEO rewrite failed: {e}")
            return AutoGEOResult(
                optimized_text=original_text,
                optimized_html=raw_html,
                original_text=original_text,
                rules_applied=[]
            )

    def optimize(
        self,
        raw_html: str,
        train_queries: List[str] = None,
        url: str = ""
    ) -> AutoGEOResult:
        """同步优化接口"""
        return asyncio.run(self.optimize_async(raw_html, train_queries, url))

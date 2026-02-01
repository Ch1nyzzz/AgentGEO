"""
插件化引用检查器架构

提供多种引用检查方式的统一接口：
- LLM: 基于 LLM 生成答案并检查引用（默认方式）
- AttrEvaluator: 使用 Attribute Evaluator 精确管道检查
- Both: 同时使用两种方式并根据策略决定最终结果
"""
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import asyncio
import logging
import re
import tempfile
from pathlib import Path

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)


@dataclass
class GEOScoreInfo:
    """GEO Score 详细信息"""
    word: float = 0.0           # 词数加权分数
    position: float = 0.0       # 位置加权分数
    wordpos: float = 0.0        # 词数+位置综合分数
    overall: float = 0.0        # 最终 GEO Score = 0.33 * (wordpos + word + pos)
    target_idx: int = 0
    num_sources: int = 0
    has_valid_citations: bool = False  # 答案是否包含有效的 [n] 格式引用


class CitationMethod(str, Enum):
    """引用检查方式枚举"""
    LLM = "llm"
    ATTR_EVALUATOR = "attr_evaluator"
    BOTH = "both"


@dataclass
class CitationResult:
    """统一的引用检查结果"""
    is_cited: bool
    generated_answer: str
    citations_found_idx: List[int]
    method: CitationMethod = CitationMethod.LLM

    # LLM 方式特有字段
    cited_by_index: bool = False
    cited_by_url: bool = False

    # Attribute Evaluator 方式特有字段
    attr_citations: Optional[List[int]] = None  # [1,0,1,...] 格式
    highlight_set: Optional[List[Dict]] = None

    # 复合结果（both 模式）
    llm_result: Optional["CitationResult"] = None
    attr_result: Optional["CitationResult"] = None

    # GEO Score（V2.3 新增）
    geo_score: Optional[GEOScoreInfo] = None


def compute_geo_score(
    generated_answer: str,
    target_idx: int,
    num_sources: int
) -> GEOScoreInfo:
    """
    计算 GEO Score

    Args:
        generated_answer: LLM 生成的包含 [n] 格式引用的答案
        target_idx: 目标文档索引（1-based）
        num_sources: 总文档数量

    Returns:
        GEOScoreInfo: GEO Score 详细信息
    """
    from AutoGEO.autogeo.evaluation.metrics.geo_score import (
        extract_citations_new,
        impression_word_count_simple,
        impression_pos_count_simple,
        impression_wordpos_count_simple,
    )

    # 检查是否有有效的 [n] 格式引用
    citation_pattern = r'\[[^\w\s]*\d+[^\w\s]*\]'
    has_valid_citations = bool(re.search(citation_pattern, generated_answer))

    if not has_valid_citations or not generated_answer.strip():
        return GEOScoreInfo(
            word=0.0,
            position=0.0,
            wordpos=0.0,
            overall=0.0,
            target_idx=target_idx,
            num_sources=num_sources,
            has_valid_citations=False,
        )

    try:
        # 提取引用结构
        sentences = extract_citations_new(generated_answer)

        # 计算三种 impression score
        word_scores = impression_word_count_simple(sentences, n=num_sources, normalize=True)
        pos_scores = impression_pos_count_simple(sentences, n=num_sources, normalize=True)
        wordpos_scores = impression_wordpos_count_simple(sentences, n=num_sources, normalize=True)

        # 获取目标文档的分数（target_idx 是 1-based，转为 0-based）
        idx = target_idx - 1
        if 0 <= idx < len(word_scores):
            word_score = word_scores[idx]
            pos_score = pos_scores[idx]
            wordpos_score = wordpos_scores[idx]
        else:
            word_score = 0.0
            pos_score = 0.0
            wordpos_score = 0.0

        # 计算综合 GEO Score
        overall = (wordpos_score + word_score + pos_score) / 3.0

        return GEOScoreInfo(
            word=word_score,
            position=pos_score,
            wordpos=wordpos_score,
            overall=overall,
            target_idx=target_idx,
            num_sources=num_sources,
            has_valid_citations=True,
        )
    except Exception as e:
        logger.warning(f"Failed to compute GEO Score: {e}")
        return GEOScoreInfo(
            word=0.0,
            position=0.0,
            wordpos=0.0,
            overall=0.0,
            target_idx=target_idx,
            num_sources=num_sources,
            has_valid_citations=has_valid_citations,
        )


def compute_geo_score_from_citations(
    citations: List[int],
    target_idx: int,
    num_sources: int
) -> GEOScoreInfo:
    """
    根据二进制 citations 列表计算简化版 GEO Score

    当没有标准 [n] 格式引用时，使用 AttrEvaluator 返回的 citations 列表
    这是一个简化版本，只能计算是否被引用，无法计算位置和词数加权

    Args:
        citations: 二进制引用列表 [0,1,0,...] 表示第 i 个文档是否被引用
        target_idx: 目标文档索引（1-based）
        num_sources: 总文档数量

    Returns:
        GEOScoreInfo: 简化版 GEO Score
    """
    # 计算被引用的文档总数
    cited_count = sum(citations) if citations else 0

    # 检查目标文档是否被引用
    idx = target_idx - 1
    is_target_cited = (0 <= idx < len(citations) and citations[idx] == 1)

    if not is_target_cited or cited_count == 0:
        return GEOScoreInfo(
            word=0.0,
            position=0.0,
            wordpos=0.0,
            overall=0.0,
            target_idx=target_idx,
            num_sources=num_sources,
            has_valid_citations=False,
        )

    # 简化版：假设所有被引用的文档平均分配分数
    score = 1.0 / cited_count

    return GEOScoreInfo(
        word=score,
        position=score,
        wordpos=score,
        overall=score,
        target_idx=target_idx,
        num_sources=num_sources,
        has_valid_citations=True,  # 虽然没有 [n] 格式，但有有效引用
    )


class BaseCitationChecker(ABC):
    """引用检查器抽象基类"""

    @abstractmethod
    async def check(
        self,
        query: str,
        target_content: str,
        target_url: str,
        target_idx: int,
        retrieved_docs: List[Any],
        competitor_contents: List[str],
    ) -> CitationResult:
        """
        检查目标内容是否被引用

        Args:
            query: 查询问题
            target_content: 目标文档内容
            target_url: 目标文档 URL
            target_idx: 目标文档在结果列表中的索引（1-based）
            retrieved_docs: 检索到的竞争文档列表
            competitor_contents: 竞争文档内容列表

        Returns:
            CitationResult: 引用检查结果
        """
        pass


class LLMCitationChecker(BaseCitationChecker):
    """LLM 引用检查器（当前实现）"""

    def __init__(self, llm: "BaseChatModel", max_snippet_length: int = 2000):
        self.llm = llm
        self.max_snippet_length = max_snippet_length

    async def _ainvoke(self, prompt_input):
        if hasattr(self.llm, "ainvoke"):
            return await self.llm.ainvoke(prompt_input)
        return await asyncio.to_thread(self.llm.invoke, prompt_input)

    async def check(
        self,
        query: str,
        target_content: str,
        target_url: str,
        target_idx: int,
        retrieved_docs: List[Any],
        competitor_contents: List[str],
    ) -> CitationResult:
        from langchain_core.prompts import ChatPromptTemplate

        # 构建 sources 文本
        sources_text = ""
        for i, content in enumerate(competitor_contents):
            if not content:
                continue
            content = content[: self.max_snippet_length]
            sources_text += f"[Source {i + 1}] \nContent: {content}\n\n"

        # 添加目标文档
        if not target_content or not target_content.strip():
            logger.warning("target_content is empty, returning not cited")
            return CitationResult(
                is_cited=False,
                generated_answer="",
                citations_found_idx=[],
                method=CitationMethod.LLM,
            )

        target_content_trimmed = target_content[: self.max_snippet_length]
        sources_text += f"[Source {target_idx}] \nContent: {target_content_trimmed}\n"

        messages = [
            (
                "system",
                "Write an accurate and concise answer for the given user question, "
                "using _only_ the provided summarized web search results. The answer "
                "should be correct, high-quality, and written by an expert using an "
                "unbiased and journalistic tone. The user's language of choice such as "
                "English, Francais, Espanol, Deutsch, or Japanese should be used. The "
                "answer should be informative, interesting, and engaging. The answer's "
                "logic and reasoning should be rigorous and defensible. Every sentence "
                "in the answer should be _immediately followed_ by an in-line citation "
                "to the search result(s). The cited search result(s) should fully support "
                "_all_ the information in the sentence. Search results need to be cited "
                "using [index]. When citing several search results, use [1][2][3] format "
                "rather than [1, 2, 3]. You can use multiple search results to respond "
                "comprehensively while avoiding irrelevant search results.",
            ),
            (
                "human",
                "Question: {query}\n\nSearch Results:\n{sources}",
            ),
        ]

        prompt = ChatPromptTemplate.from_messages(messages)

        try:
            response = await self._ainvoke(prompt.format(query=query, sources=sources_text))
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            raise RuntimeError("LLM generation failed.") from exc

        answer = response.content

        # 检查引用
        cited_by_index = f"[{target_idx}]" in answer
        cited_by_url = bool(target_url) and target_url in answer
        is_cited = cited_by_index or cited_by_url
        cited_indices = list({int(m) for m in re.findall(r"\[(\d+)\]", answer)})

        # 计算 GEO Score（LLM 答案有标准 [n] 格式）
        num_sources = len(competitor_contents) + 1  # 竞争文档 + 目标文档
        geo_score_info = compute_geo_score(answer, target_idx, num_sources)

        return CitationResult(
            is_cited=is_cited,
            generated_answer=answer,
            citations_found_idx=cited_indices,
            method=CitationMethod.LLM,
            cited_by_index=cited_by_index,
            cited_by_url=cited_by_url,
            geo_score=geo_score_info,
        )


class AttrEvaluatorCitationChecker(BaseCitationChecker):
    """Attribute Evaluator 引用检查器"""

    def __init__(self, config_file: Optional[str] = None, use_fast_mode: bool = True):
        self.config_file = config_file
        self.use_fast_mode = use_fast_mode

    async def check(
        self,
        query: str,
        target_content: str,
        target_url: str,
        target_idx: int,
        retrieved_docs: List[Any],
        competitor_contents: List[str],
    ) -> CitationResult:
        # 构建文档列表：竞争文档 + 目标文档（插入到 target_idx-1 位置）
        all_docs = list(competitor_contents)
        insert_pos = target_idx - 1
        if insert_pos < 0:
            insert_pos = 0
        if insert_pos > len(all_docs):
            insert_pos = len(all_docs)
        all_docs.insert(insert_pos, target_content)
        
        # 输出搜索结果信息
        logger.info(f"\n=== AttrEvaluatorCitationChecker 开始检查引用 ===")
        
        try:
            if self.use_fast_mode:
                logger.info(f"使用 fast_return_res 进行引用检查")
                from src.attr_evaluator.fast_return_res import fast_return_res
                # 调用 fast_return_res（在线程中运行以避免阻塞）
                result = await asyncio.to_thread(
                    fast_return_res, "test_id", query, all_docs, self.config_file
                )
            else:
                logger.info(f"使用原始 return_res 进行引用检查")
                from src.attr_evaluator.run_dataset import return_res, pre_init
                # 初始化 args
                arg_list = []
                if self.config_file:
                    arg_list.extend(["--config-file", self.config_file])
                args = pre_init(arg_list)
                # 调用 return_res（在线程中运行以避免阻塞）
                result = await asyncio.to_thread(
                    return_res, "test_id", query, all_docs, args
                )
        except Exception as e:
            logger.error("AttrEvaluator check failed: %s", e)
            import traceback
            logger.error("Traceback: %s", traceback.format_exc())
            return CitationResult(
                is_cited=False,
                generated_answer="",
                citations_found_idx=[],
                method=CitationMethod.ATTR_EVALUATOR,
                attr_citations=[],
                highlight_set=[],
            )

        citations = result.get("citations", [])
        logger.info(f"引用结果: {citations}")
        
        # 检查目标文档是否被引用（target_idx 是 1-based）
        is_cited = False
        if target_idx <= len(citations):
            is_cited = citations[target_idx - 1] == 1
            logger.info(f"目标文档是否被引用: {'✅ 是' if is_cited else '❌ 否'}")

        # 收集所有被引用的文档索引（转换为 1-based）
        citations_found_idx = [i + 1 for i, c in enumerate(citations) if c == 1]
        logger.info(f"所有被引用的文档索引 (1-based): {citations_found_idx}")
        
        generated_answer = result.get("answer_prompt", "")
        # logger.info(f"生成的答案: {generated_answer[:100]}...")
        # logger.info(f"高亮集合大小: {len(result.get('highlight_set', []))}")
        logger.info(f"=== AttrEvaluatorCitationChecker 引用检查结束 ===")

        # 计算 GEO Score
        num_sources = len(all_docs)
        # 检查生成的答案是否有 [n] 格式引用
        citation_pattern = r'\[[^\w\s]*\d+[^\w\s]*\]'
        has_citation_format = bool(re.search(citation_pattern, generated_answer))

        if has_citation_format:
            # 使用标准方法计算
            geo_score_info = compute_geo_score(generated_answer, target_idx, num_sources)
        else:
            # 使用简化版（基于 citations 列表）
            geo_score_info = compute_geo_score_from_citations(citations, target_idx, num_sources)

        return CitationResult(
            is_cited=is_cited,
            generated_answer=generated_answer,
            citations_found_idx=citations_found_idx,
            method=CitationMethod.ATTR_EVALUATOR,
            attr_citations=citations,
            highlight_set=result.get("highlight_set", []),
            geo_score=geo_score_info,
        )


class CompositeCitationChecker(BaseCitationChecker):
    """
    复合引用检查器（同时使用两种方式）

    策略说明：
    - any: 任一方式检测到引用即认为被引用
    - all: 两种方式都检测到引用才认为被引用
    - llm_primary: 以 LLM 结果为主，attr 结果作为补充信息
    - attr_primary: 以 AttrEvaluator 结果为主，LLM 结果作为补充信息
    """

    def __init__(
        self,
        llm_checker: LLMCitationChecker,
        attr_checker: AttrEvaluatorCitationChecker,
        strategy: str = "any",
    ):
        self.llm_checker = llm_checker
        self.attr_checker = attr_checker
        self.strategy = strategy

    async def check(
        self,
        query: str,
        target_content: str,
        target_url: str,
        target_idx: int,
        retrieved_docs: List[Any],
        competitor_contents: List[str],
    ) -> CitationResult:
        # 并行执行两种检查
        llm_task = self.llm_checker.check(
            query, target_content, target_url, target_idx,
            retrieved_docs, competitor_contents
        )
        attr_task = self.attr_checker.check(
            query, target_content, target_url, target_idx,
            retrieved_docs, competitor_contents
        )

        llm_result, attr_result = await asyncio.gather(
            llm_task, attr_task, return_exceptions=True
        )

        # 处理异常情况
        if isinstance(llm_result, Exception):
            logger.error("LLM checker failed: %s", llm_result)
            llm_result = CitationResult(
                is_cited=False, generated_answer="", citations_found_idx=[],
                method=CitationMethod.LLM
            )
        if isinstance(attr_result, Exception):
            logger.error("AttrEvaluator checker failed: %s", attr_result)
            attr_result = CitationResult(
                is_cited=False, generated_answer="", citations_found_idx=[],
                method=CitationMethod.ATTR_EVALUATOR
            )

        # 根据策略决定最终结果
        if self.strategy == "any":
            is_cited = llm_result.is_cited or attr_result.is_cited
        elif self.strategy == "all":
            is_cited = llm_result.is_cited and attr_result.is_cited
        elif self.strategy == "llm_primary":
            is_cited = llm_result.is_cited
        elif self.strategy == "attr_primary":
            is_cited = attr_result.is_cited
        else:
            logger.warning("Unknown strategy '%s', defaulting to 'any'", self.strategy)
            is_cited = llm_result.is_cited or attr_result.is_cited

        # 合并引用索引
        all_citations = set(llm_result.citations_found_idx) | set(attr_result.citations_found_idx)

        # 优先使用 LLM 生成的答案
        generated_answer = llm_result.generated_answer or attr_result.generated_answer

        # 根据策略合并 GEO Score
        llm_geo = llm_result.geo_score
        attr_geo = attr_result.geo_score

        if self.strategy == "llm_primary":
            merged_geo_score = llm_geo
        elif self.strategy == "attr_primary":
            merged_geo_score = attr_geo
        elif self.strategy in ("any", "all"):
            # 优先使用有有效引用的那个，如果都有则取平均
            if llm_geo and attr_geo and llm_geo.has_valid_citations and attr_geo.has_valid_citations:
                merged_geo_score = GEOScoreInfo(
                    word=(llm_geo.word + attr_geo.word) / 2,
                    position=(llm_geo.position + attr_geo.position) / 2,
                    wordpos=(llm_geo.wordpos + attr_geo.wordpos) / 2,
                    overall=(llm_geo.overall + attr_geo.overall) / 2,
                    target_idx=llm_geo.target_idx,
                    num_sources=llm_geo.num_sources,
                    has_valid_citations=True,
                )
            elif llm_geo and llm_geo.has_valid_citations:
                merged_geo_score = llm_geo
            elif attr_geo and attr_geo.has_valid_citations:
                merged_geo_score = attr_geo
            else:
                merged_geo_score = llm_geo or attr_geo
        else:
            merged_geo_score = llm_geo or attr_geo

        return CitationResult(
            is_cited=is_cited,
            generated_answer=generated_answer,
            citations_found_idx=sorted(all_citations),
            method=CitationMethod.BOTH,
            cited_by_index=llm_result.cited_by_index,
            cited_by_url=llm_result.cited_by_url,
            attr_citations=attr_result.attr_citations,
            highlight_set=attr_result.highlight_set,
            llm_result=llm_result,
            attr_result=attr_result,
            geo_score=merged_geo_score,
        )


def create_citation_checker(
    method: CitationMethod,
    llm: Optional["BaseChatModel"] = None,
    max_snippet_length: int = 2000,
    attr_evaluator_config: Optional[str] = None,
    composite_strategy: str = "any",
    use_fast_mode: bool = True,
) -> BaseCitationChecker:
    """
    工厂函数：创建引用检查器

    Args:
        method: 引用检查方式
        llm: LLM 实例（LLM 和 BOTH 模式需要）
        max_snippet_length: 内容片段最大长度
        attr_evaluator_config: AttrEvaluator 配置文件路径
        composite_strategy: 复合模式策略
        use_fast_mode: 是否使用快速模式，默认为True

    Returns:
        BaseCitationChecker: 引用检查器实例
    """
    if method == CitationMethod.LLM:
        if llm is None:
            raise ValueError("LLM instance is required for LLM citation method")
        return LLMCitationChecker(llm, max_snippet_length)

    elif method == CitationMethod.ATTR_EVALUATOR:
        return AttrEvaluatorCitationChecker(attr_evaluator_config, use_fast_mode)

    elif method == CitationMethod.BOTH:
        if llm is None:
            raise ValueError("LLM instance is required for BOTH citation method")
        llm_checker = LLMCitationChecker(llm, max_snippet_length)
        attr_checker = AttrEvaluatorCitationChecker(attr_evaluator_config, use_fast_mode)
        return CompositeCitationChecker(llm_checker, attr_checker, composite_strategy)

    else:
        raise ValueError(f"Unknown citation method: {method}")

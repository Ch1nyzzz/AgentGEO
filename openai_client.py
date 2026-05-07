import re

import nltk
from openai import AsyncOpenAI

from autogeo.evaluation.metrics import (
    extract_citations_new,
    impression_pos_count_simple,
    impression_word_count_simple,
    impression_wordpos_count_simple,
)


DEFAULT_MODEL = "gpt-5-mini"

SYSTEM_PROMPT = (
    "You are a research assistant. The user will provide a question and a numbered "
    "list of source URLs. Read the provided sources and answer using only those "
    "sources.\n\n"
    "Rules:\n"
    "1. Use only information from the provided sources.\n"
    "2. Cite only sources that directly support the sentence.\n"
    "3. Use [index] citations such as [1] or [1][3].\n"
    "4. Do not cite URLs, markdown links, or internal browsing identifiers.\n"
    "5. Keep the answer accurate, concise, and neutral."
)


def build_sources(
    target_url: str,
    competitor_urls: list[str],
    target_position: int | None = None,
) -> tuple[str, int]:
    urls = [url for url in competitor_urls if url]
    target_idx = len(urls) + 1 if target_position is None else target_position
    if not 1 <= target_idx <= len(urls) + 1:
        raise ValueError(f"target_position must be in [1, {len(urls) + 1}]")

    urls.insert(target_idx - 1, target_url)
    return "".join(f"[Source {i + 1}] URL: {url}\n" for i, url in enumerate(urls)), target_idx


def compute_geo_scores(answer: str, target_idx: int, num_sources: int) -> dict[str, float]:
    target_id = target_idx - 1
    try:
        citations = extract_citations_new(answer)
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        citations = extract_citations_new(answer)

    return {
        "wordpos": impression_wordpos_count_simple(citations, n=num_sources)[target_id],
        "word": impression_word_count_simple(citations, n=num_sources)[target_id],
        "pos": impression_pos_count_simple(citations, n=num_sources)[target_id],
    }


def score_answer(
    answer: str,
    target_idx: int,
    target_url: str = "",
    num_sources: int = 0,
) -> dict:
    cited_by_index = f"[{target_idx}]" in answer
    cited_by_url = bool(target_url) and target_url in answer
    cited_indices = sorted({int(match) for match in re.findall(r"\[(\d+)\]", answer)})
    num_sources = num_sources or max(cited_indices + [target_idx])

    return {
        "cr": 1.0 if cited_by_index or cited_by_url else 0.0,
        **compute_geo_scores(answer, target_idx, num_sources),
        "is_cited": cited_by_index or cited_by_url,
        "cited_by_index": cited_by_index,
        "cited_by_url": cited_by_url,
        "cited_indices": cited_indices,
        "target_idx": target_idx,
        "num_sources": num_sources,
    }


def extract_output_text(response) -> str:
    output_text = getattr(response, "output_text", "") or ""
    if output_text:
        return output_text

    parts = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", None) == "output_text":
                parts.append(content.text)
    return "".join(parts)


class OpenAIClient:
    def __init__(self, api_key: str | None = None):
        self.client = AsyncOpenAI(api_key=api_key)

    async def score(
        self,
        query: str,
        target_url: str,
        competitor_urls: list[str],
        model: str = DEFAULT_MODEL,
        target_position: int | None = None,
        max_output_tokens: int = 8000,
    ) -> dict:
        competitor_urls = [url for url in competitor_urls if url]
        sources_text, target_idx = build_sources(target_url, competitor_urls, target_position)
        user_msg = (
            f"Question: {query}\n\n"
            "Below are the ONLY sources you may use. Visit each URL, read its content, "
            "and answer with [index] citations.\n\n"
            f"Sources:\n{sources_text}"
        )

        response = await self.client.responses.create(
            model=model,
            instructions=SYSTEM_PROMPT,
            input=user_msg,
            tools=[{"type": "web_search_preview"}],
            reasoning={"effort": "low"},
            text={"verbosity": "low"},
            max_output_tokens=max_output_tokens,
        )
        answer = extract_output_text(response)
        return {
            **score_answer(answer, target_idx, target_url, len(competitor_urls) + 1),
            "generated_answer": answer,
            "target_url": target_url,
            "competitor_urls": competitor_urls,
            "model": model,
        }


async def score(
    query: str,
    target_url: str,
    competitor_urls: list[str],
    model: str = DEFAULT_MODEL,
    target_position: int | None = None,
    max_output_tokens: int = 8000,
) -> dict:
    return await OpenAIClient().score(
        query=query,
        target_url=target_url,
        competitor_urls=competitor_urls,
        model=model,
        target_position=target_position,
        max_output_tokens=max_output_tokens,
    )

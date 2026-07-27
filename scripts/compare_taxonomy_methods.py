"""
对比 Semantic Alignment vs Content Quality 的两种分类方法：
1. LLM taxonomy (已有结果 - 基于 explainer prompt 的主观判断)
2. Answer-evidence matching (新方法 - 基于 cited doc 中的 answer 是否存在于 uncited doc)

两个文件 949 条逐条对应，可做精确一致性对比。
"""

import asyncio
import json
import os
import re
from collections import Counter
from openai import AsyncOpenAI
from dotenv import load_dotenv

# ── Paths ──
LLM_RESULTS_PATH = "/Users/erv1n/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/3100c5c43e308c23b1cf67f6e92ad0e9/Message/MessageTemp/9e20f478899dc29eb19741386f9343c8/OpenData/2/f2b82df822ceedf1e42f9da2b394cc19.json"
PAIRS_PATH = "/Users/erv1n/Library/Containers/com.tencent.xinWeChat/Data/Library/Caches/com.tencent.xinWeChat/2.0b4.0.9/3100c5c43e308c23b1cf67f6e92ad0e9/SaveTemp/ab08d700a04035f840c1dc5c4877ca1d/openai_full.json"
OUTPUT_PATH = "outputs/taxonomy_comparison_results.json"

# ── Step 1: Extract answer evidence from cited doc ──
EXTRACT_ANSWER_PROMPT = """Given the following query and a cited document (which was selected by an AI engine to answer the query), extract the **answer evidence** — the specific facts, data, entities, or statements that directly answer the query.

Be concise. List only the key factual claims, not background context.

**Query:** {query}

**Cited Document:**
{cited_content}

**Answer Evidence** (bullet points of key facts that answer the query):"""

# ── Step 2: Check if answer evidence exists in uncited doc ──
CHECK_EVIDENCE_PROMPT = """Given the following answer evidence (extracted from a cited document), determine whether **semantically equivalent information** exists in the uncited document below.

"Semantically equivalent" means the same factual claims or information are present, even if worded differently. It does NOT require exact wording match.

**Query:** {query}

**Answer Evidence (from cited document):**
{answer_evidence}

**Uncited Document (to check):**
{uncited_content}

Respond in this exact JSON format:
{{"present": true/false, "justification": "brief explanation of what was found or missing"}}"""


def parse_presence(text: str) -> tuple[bool | None, str]:
    try:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            obj = json.loads(m.group())
            return obj.get("present"), obj.get("justification", "")
    except json.JSONDecodeError:
        pass
    lower = text.lower()
    if '"present": true' in lower or '"present":true' in lower:
        return True, text
    if '"present": false' in lower or '"present":false' in lower:
        return False, text
    return None, text


async def process_pair(
    client: AsyncOpenAI, query: str, cited_content: str, uncited_content: str, semaphore: asyncio.Semaphore,
) -> dict:
    max_len = 6000
    cited_trunc = cited_content[:max_len]
    uncited_trunc = uncited_content[:max_len]

    async with semaphore:
        # Step 1: Extract answer evidence
        try:
            resp1 = await client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": EXTRACT_ANSWER_PROMPT.format(
                    query=query, cited_content=cited_trunc
                )}],
                temperature=0, max_tokens=500,
            )
            answer_evidence = resp1.choices[0].message.content
        except Exception as e:
            return {"classification": "Error", "answer_evidence": "", "present": None, "justification": str(e)}

        # Step 2: Check presence in uncited doc
        try:
            resp2 = await client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": CHECK_EVIDENCE_PROMPT.format(
                    query=query, answer_evidence=answer_evidence, uncited_content=uncited_trunc
                )}],
                temperature=0, max_tokens=300,
            )
            check_text = resp2.choices[0].message.content
            present, justification = parse_presence(check_text)
        except Exception as e:
            return {"classification": "Error", "answer_evidence": answer_evidence, "present": None, "justification": str(e)}

        if present is True:
            classification = "Content Quality"
        elif present is False:
            classification = "Semantic Alignment"
        else:
            classification = "Unknown"

        return {
            "classification": classification,
            "answer_evidence": answer_evidence,
            "present": present,
            "justification": justification,
        }


async def main():
    load_dotenv()

    with open(LLM_RESULTS_PATH, "r") as f:
        llm_data = json.load(f)
    with open(PAIRS_PATH, "r") as f:
        pairs_data = json.load(f)

    llm_results = llm_data["detailed_results"]
    assert len(pairs_data) == len(llm_results), "数据条数不匹配!"
    n = len(pairs_data)
    print(f"数据条数: {n} (逐条对应)")

    # ── 只对 SA/CQ 的条目运行新方法（TI/SE 不在判断范围内） ──
    target_categories = {"Semantic Alignment", "Content Quality"}
    target_indices = [i for i, r in enumerate(llm_results) if r["category"] in target_categories]
    print(f"LLM 分类为 SA/CQ 的条目: {len(target_indices)}")
    print(f"  Semantic Alignment: {sum(1 for i in target_indices if llm_results[i]['category'] == 'Semantic Alignment')}")
    print(f"  Content Quality: {sum(1 for i in target_indices if llm_results[i]['category'] == 'Content Quality')}")

    # 运行 answer-evidence 方法
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    semaphore = asyncio.Semaphore(20)

    print(f"\n开始运行 answer-evidence 分类 ({len(target_indices)} 条)...")
    tasks = [
        process_pair(
            client,
            pairs_data[i]["query"],
            pairs_data[i]["cited_doc"]["content"],
            pairs_data[i]["uncited_doc"]["content"],
            semaphore,
        )
        for i in target_indices
    ]
    results = await asyncio.gather(*tasks)
    print("分类完成。")

    # ── 逐条对比 ──
    agree = 0
    disagree = 0
    confusion = Counter()
    detailed = []

    for idx, result in zip(target_indices, results):
        llm_cat = llm_results[idx]["category"]
        ev_cat = result["classification"]
        is_agree = (llm_cat == ev_cat)
        if ev_cat in target_categories:
            if is_agree:
                agree += 1
            else:
                disagree += 1
        confusion[(llm_cat, ev_cat)] += 1
        detailed.append({
            "index": idx,
            "example_id": pairs_data[idx]["example_id"],
            "query": pairs_data[idx]["query"],
            "uncited_url": pairs_data[idx]["uncited_doc"]["url"],
            "llm_category": llm_cat,
            "llm_mode": llm_results[idx]["mode"],
            "evidence_category": ev_cat,
            "agree": is_agree if ev_cat in target_categories else None,
            "answer_evidence": result["answer_evidence"],
            "present_in_uncited": result["present"],
            "justification": result["justification"],
        })

    total_comparable = agree + disagree
    agreement_rate = agree / total_comparable if total_comparable > 0 else 0

    # ── 打印结果 ──
    print(f"\n{'='*60}")
    print(f"逐条一致性对比 (仅 SA/CQ)")
    print(f"{'='*60}")
    print(f"可比较条目: {total_comparable}")
    print(f"一致: {agree} ({agreement_rate*100:.1f}%)")
    print(f"不一致: {disagree} ({(1-agreement_rate)*100:.1f}%)")

    print(f"\n--- Confusion Matrix (LLM \\ Evidence) ---")
    all_cats = ["Semantic Alignment", "Content Quality", "Unknown", "Error"]
    header = f"{'LLM \\\\ Evidence':<25}" + "".join(f"{c:>22}" for c in all_cats)
    print(header)
    for llm_cat in ["Semantic Alignment", "Content Quality"]:
        row = f"{llm_cat:<25}"
        for ev_cat in all_cats:
            row += f"{confusion.get((llm_cat, ev_cat), 0):>22}"
        print(row)

    print(f"\n--- 不一致分析 ---")
    print(f"LLM=SA, Evidence=CQ (LLM认为缺信息, 新方法认为信息存在): {confusion.get(('Semantic Alignment', 'Content Quality'), 0)}")
    print(f"LLM=CQ, Evidence=SA (LLM认为质量差, 新方法认为缺信息): {confusion.get(('Content Quality', 'Semantic Alignment'), 0)}")

    # 新方法的总体分布
    ev_counter = Counter(r["classification"] for r in results)
    print(f"\n--- Answer-Evidence 方法总体分布 ---")
    for cat, cnt in ev_counter.most_common():
        print(f"  {cat}: {cnt} ({cnt/len(results)*100:.1f}%)")

    # ── 保存 ──
    output = {
        "summary": {
            "total_sa_cq": len(target_indices),
            "comparable": total_comparable,
            "agreement": agree,
            "agreement_rate": agreement_rate,
            "confusion_matrix": {f"{k[0]} -> {k[1]}": v for k, v in sorted(confusion.items())},
            "evidence_distribution": {cat: cnt for cat, cnt in ev_counter.most_common()},
            "llm_distribution": {
                "Semantic Alignment": sum(1 for i in target_indices if llm_results[i]["category"] == "Semantic Alignment"),
                "Content Quality": sum(1 for i in target_indices if llm_results[i]["category"] == "Content Quality"),
            },
        },
        "detailed_results": detailed,
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存到: {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())

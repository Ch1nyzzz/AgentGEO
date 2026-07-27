"""
对全部 949 条 pair 用 evidence-based 顺序诊断流程重新分类（只输出四个大类，不分 subcategory）。
然后和 LLM taxonomy 结果做逐条一致性对比。
"""

import asyncio
import json
import os
import re
from collections import Counter
from openai import AsyncOpenAI
from dotenv import load_dotenv

LLM_RESULTS_PATH = "/Users/erv1n/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/3100c5c43e308c23b1cf67f6e92ad0e9/Message/MessageTemp/9e20f478899dc29eb19741386f9343c8/OpenData/2/f2b82df822ceedf1e42f9da2b394cc19.json"
RAW_PAIRS_PATH = "/Users/erv1n/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/3100c5c43e308c23b1cf67f6e92ad0e9/Message/MessageTemp/9e20f478899dc29eb19741386f9343c8/File/openai_full.json"
OUTPUT_PATH = "outputs/evidence_method_results_full.json"
REPORT_PATH = "outputs/taxonomy_comparison_report_full.md"

# ── Step 1: Extract answer evidence ──
EXTRACT_ANSWER_PROMPT = """Given the following query and a cited document (which was selected by an AI engine to answer the query), extract the **answer evidence** — the specific facts, data, entities, or statements that directly answer the query.

Be concise. List only the key factual claims, not background context.

**Query:** {query}

**Cited Document:**
{cited_content}

**Answer Evidence** (bullet points of key facts that answer the query):"""

# ── Step 2: Full sequential diagnosis (四大类, 无 subcategory) ──
DIAGNOSE_PROMPT = """You are classifying why an uncited document was NOT cited by a generative AI engine. Follow these steps IN ORDER. Stop at the first step that applies.

**Step 1 — Technical Integrity**
Is the uncited document technically broken, empty, inaccessible, or overwhelmed by noise (ads, boilerplate, garbled text, login walls, JS rendering failures)?
→ If YES: classify as **Technical Integrity**. STOP.

**Step 2 — Semantic Alignment**
Using the answer evidence below (extracted from the cited document), check: does semantically equivalent information exist in the uncited document? "Semantically equivalent" means the same factual claims are present, even if worded differently.
IMPORTANT: If the uncited document contains **partial, fragmented, or obscured** versions of the answer evidence (e.g., the key facts are buried in noise, scattered across sections, or incomplete but partially present), do NOT classify as Semantic Alignment. Instead, pass it to Step 3 (Content Quality).
Only classify as Semantic Alignment if the answer evidence is **clearly and entirely absent** — the document simply does not contain the needed information at all.
→ If the answer evidence is clearly and entirely NOT present: classify as **Semantic Alignment**. STOP.

**Step 3 — Content Quality**
The answer evidence exists in the uncited document, but it wasn't cited. Is the information poorly presented (unstructured, too verbose, fragmented, buried in noise)?
→ If YES: classify as **Content Quality**. STOP.

**Step 4 — Systemic Exclusion**
The document is readable, contains the answer, and presents it well — yet it still wasn't cited. It was likely outranked by a higher-authority source or truncated by the context window.
→ Classify as **Systemic Exclusion**.

---

**Query:** {query}

**Answer Evidence (from cited document):**
{answer_evidence}

**Uncited Document:**
{uncited_content}

Respond in this exact JSON format:
{{"step": 1/2/3/4, "category": "Technical Integrity" or "Semantic Alignment" or "Content Quality" or "Systemic Exclusion", "justification": "brief explanation"}}"""


def parse_diagnosis(text: str) -> dict:
    try:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            return json.loads(m.group())
    except json.JSONDecodeError:
        pass
    return {"step": 0, "category": "Unknown", "justification": text}


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
            return {"category": "Error", "answer_evidence": "", "justification": str(e)}

        # Step 2: Sequential diagnosis
        try:
            resp2 = await client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": DIAGNOSE_PROMPT.format(
                    query=query, answer_evidence=answer_evidence, uncited_content=uncited_trunc
                )}],
                temperature=0, max_tokens=300,
            )
            diag = parse_diagnosis(resp2.choices[0].message.content)
        except Exception as e:
            return {"category": "Error", "answer_evidence": answer_evidence, "justification": str(e)}

        return {
            "category": diag.get("category", "Unknown"),
            "answer_evidence": answer_evidence,
            "justification": diag.get("justification", ""),
        }


def generate_report(llm_data, evidence_results, pairs_data):
    """生成对比报告"""
    llm_dr = llm_data["detailed_results"]
    n = len(pairs_data)

    # 四大类列表
    CATS = ["Technical Integrity", "Semantic Alignment", "Content Quality", "Systemic Exclusion"]

    # ── 1. 各自分布 ──
    ev_counter = Counter(r["category"] for r in evidence_results)
    llm_counter = Counter(r["category"] for r in llm_dr)
    # 合并 Window Truncation 到 Systemic Exclusion
    llm_counter["Systemic Exclusion"] += llm_counter.pop("Window Truncation", 0)

    # ── 2. 逐条一致性 ──
    # 同样合并 LLM 的 Window Truncation
    agree = 0
    disagree = 0
    confusion = Counter()
    disagree_details = []

    for i in range(n):
        llm_cat = llm_dr[i]["category"]
        if llm_cat == "Window Truncation":
            llm_cat = "Systemic Exclusion"
        ev_cat = evidence_results[i]["category"]
        if ev_cat in CATS and llm_cat in CATS:
            confusion[(llm_cat, ev_cat)] += 1
            if llm_cat == ev_cat:
                agree += 1
            else:
                disagree += 1
                disagree_details.append({
                    "index": i,
                    "example_id": pairs_data[i]["example_id"],
                    "query": pairs_data[i]["query"],
                    "llm_category": llm_cat,
                    "evidence_category": ev_cat,
                    "justification": evidence_results[i]["justification"],
                })

    total_comparable = agree + disagree
    agreement_rate = agree / total_comparable if total_comparable > 0 else 0

    # Cohen's Kappa
    import numpy as np
    cat_to_idx = {c: i for i, c in enumerate(CATS)}
    cm = np.zeros((len(CATS), len(CATS)), dtype=int)
    for (lc, ec), cnt in confusion.items():
        if lc in cat_to_idx and ec in cat_to_idx:
            cm[cat_to_idx[lc], cat_to_idx[ec]] = cnt
    po = np.trace(cm) / cm.sum() if cm.sum() > 0 else 0
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    pe = (row_sums * col_sums).sum() / (cm.sum() ** 2) if cm.sum() > 0 else 0
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0

    # ── 打印 ──
    print(f"\n{'='*60}")
    print("Full Evidence-Based vs LLM Taxonomy Comparison")
    print(f"{'='*60}")

    print(f"\n--- Distribution ---")
    print(f"{'Category':<25} {'LLM':>10} {'Evidence':>10}")
    for cat in CATS:
        print(f"{cat:<25} {llm_counter.get(cat, 0):>10} {ev_counter.get(cat, 0):>10}")

    print(f"\n--- Agreement ---")
    print(f"Comparable: {total_comparable}")
    print(f"Agreement: {agree} ({agreement_rate*100:.1f}%)")
    print(f"Cohen's Kappa: {kappa:.3f}")

    print(f"\n--- Confusion Matrix (LLM \\ Evidence) ---")
    header = f"{'':.<25}" + "".join(f"{c:>15}" for c in CATS)
    print(header)
    for lc in CATS:
        row = f"{lc:.<25}"
        for ec in CATS:
            row += f"{confusion.get((lc, ec), 0):>15}"
        print(row)

    # ── 写报告 ──
    report = f"""# Evidence-Based vs LLM Taxonomy — Full Comparison Report

## Overview

- Total pairs: {n}
- Comparable (both valid): {total_comparable}
- Agreement: {agree} ({agreement_rate*100:.1f}%)
- Cohen's Kappa: {kappa:.3f}

## Distribution Comparison

| Category | LLM Taxonomy | Evidence Method |
|---|---|---|
"""
    for cat in CATS:
        lc = llm_counter.get(cat, 0)
        ec = ev_counter.get(cat, 0)
        report += f"| {cat} | {lc} ({lc/n*100:.1f}%) | {ec} ({ec/n*100:.1f}%) |\n"

    report += f"""
## Confusion Matrix (LLM \\ Evidence)

| | Technical Integrity | Semantic Alignment | Content Quality | Systemic Exclusion |
|---|---|---|---|---|
"""
    for lc in CATS:
        row = f"| **{lc}** |"
        for ec in CATS:
            v = confusion.get((lc, ec), 0)
            row += f" {v} |"
        report += row + "\n"

    report += f"""
## Agreement Analysis

- **Overall agreement**: {agreement_rate*100:.1f}%
- **Cohen's Kappa**: {kappa:.3f}

### Per-category agreement
"""
    for cat in CATS:
        cat_total = sum(confusion.get((cat, ec), 0) for ec in CATS)
        cat_agree = confusion.get((cat, cat), 0)
        cat_rate = cat_agree / cat_total * 100 if cat_total > 0 else 0
        report += f"- **{cat}**: {cat_agree}/{cat_total} ({cat_rate:.1f}%)\n"

    report += f"""
### Key disagreements
- LLM=Content Quality → Evidence=Semantic Alignment: {confusion.get(('Content Quality', 'Semantic Alignment'), 0)}
- LLM=Content Quality → Evidence=Technical Integrity: {confusion.get(('Content Quality', 'Technical Integrity'), 0)}
- LLM=Semantic Alignment → Evidence=Content Quality: {confusion.get(('Semantic Alignment', 'Content Quality'), 0)}
- LLM=Semantic Alignment → Evidence=Technical Integrity: {confusion.get(('Semantic Alignment', 'Technical Integrity'), 0)}
- LLM=Technical Integrity → Evidence=Semantic Alignment: {confusion.get(('Technical Integrity', 'Semantic Alignment'), 0)}
"""

    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"\n报告已保存到: {REPORT_PATH}")

    return disagree_details


async def main():
    load_dotenv()

    with open(RAW_PAIRS_PATH, "r") as f:
        pairs_data = json.load(f)
    with open(LLM_RESULTS_PATH, "r") as f:
        llm_data = json.load(f)

    n = len(pairs_data)
    print(f"总条数: {n}")

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    semaphore = asyncio.Semaphore(256)

    print(f"开始全量 evidence-based 分类 ({n} 条)...")
    tasks = [
        process_pair(client, p["query"], p["cited_doc"]["content"], p["uncited_doc"]["content"], semaphore)
        for p in pairs_data
    ]
    results = await asyncio.gather(*tasks)
    print("分类完成。")

    # ── 保存结果（和 LLM 格式一致，mode 留空） ──
    cat_counter = Counter(r["category"] for r in results)
    total = len(results)
    cat_stats = {cat: {"count": cnt, "ratio": cnt / total} for cat, cnt in cat_counter.most_common()}

    detailed = []
    for p, r in zip(pairs_data, results):
        detailed.append({
            "example_id": p["example_id"],
            "query": p["query"],
            "category": r["category"],
            "mode": "",
            "answer_evidence": r["answer_evidence"],
            "justification": r["justification"],
        })

    output = {
        "total": total,
        "category_stats": cat_stats,
        "mode_stats": {},
        "detailed_results": detailed,
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"结果已保存到: {OUTPUT_PATH}")

    # ── 生成对比报告 ──
    generate_report(llm_data, results, pairs_data)


if __name__ == "__main__":
    asyncio.run(main())

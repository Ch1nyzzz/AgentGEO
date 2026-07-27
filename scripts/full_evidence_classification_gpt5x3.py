"""
用 GPT-5 跑三次 evidence-based 分类，对比：
1. 三次之间的互相一致性 (inter-run agreement)
2. 每次和原始 LLM taxonomy 的一致性
"""

import asyncio
import json
import os
import re
from collections import Counter
from openai import AsyncOpenAI
from dotenv import load_dotenv
import numpy as np

LLM_RESULTS_PATH = "/Users/erv1n/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/3100c5c43e308c23b1cf67f6e92ad0e9/Message/MessageTemp/9e20f478899dc29eb19741386f9343c8/OpenData/2/f2b82df822ceedf1e42f9da2b394cc19.json"
RAW_PAIRS_PATH = "/Users/erv1n/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/3100c5c43e308c23b1cf67f6e92ad0e9/Message/MessageTemp/9e20f478899dc29eb19741386f9343c8/File/openai_full.json"
OUTPUT_DIR = "outputs/gpt5_runs"

MODEL = "gpt-5"
CONCURRENCY = 256
NUM_RUNS = 3
CATS = ["Technical Integrity", "Semantic Alignment", "Content Quality", "Systemic Exclusion"]

EXTRACT_ANSWER_PROMPT = """Given the following query and a cited document (which was selected by an AI engine to answer the query), extract the **answer evidence** — the specific facts, data, entities, or statements that directly answer the query.

Be concise. List only the key factual claims, not background context.

**Query:** {query}

**Cited Document:**
{cited_content}

**Answer Evidence** (bullet points of key facts that answer the query):"""

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


async def process_pair(client, query, cited_content, uncited_content, semaphore):
    max_len = 6000
    cited_trunc = cited_content[:max_len]
    uncited_trunc = uncited_content[:max_len]

    async with semaphore:
        try:
            resp1 = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": EXTRACT_ANSWER_PROMPT.format(
                    query=query, cited_content=cited_trunc
                )}],
                max_completion_tokens=8000,
            )
            answer_evidence = resp1.choices[0].message.content
        except Exception as e:
            return {"category": "Error", "answer_evidence": "", "justification": str(e)}

        try:
            resp2 = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": DIAGNOSE_PROMPT.format(
                    query=query, answer_evidence=answer_evidence, uncited_content=uncited_trunc
                )}],
                max_completion_tokens=4000,
            )
            diag = parse_diagnosis(resp2.choices[0].message.content)
        except Exception as e:
            return {"category": "Error", "answer_evidence": answer_evidence, "justification": str(e)}

        return {
            "category": diag.get("category", "Unknown"),
            "answer_evidence": answer_evidence,
            "justification": diag.get("justification", ""),
            "raw_diagnosis": resp2.choices[0].message.content,
        }


def compute_kappa(cats_a, cats_b):
    """计算两组分类结果之间的 Cohen's Kappa"""
    cat_to_idx = {c: i for i, c in enumerate(CATS)}
    cm = np.zeros((len(CATS), len(CATS)), dtype=int)
    for a, b in zip(cats_a, cats_b):
        if a in cat_to_idx and b in cat_to_idx:
            cm[cat_to_idx[a], cat_to_idx[b]] += 1
    total = cm.sum()
    if total == 0:
        return 0, 0, cm
    po = np.trace(cm) / total
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    pe = (row_sums * col_sums).sum() / (total ** 2)
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0
    return po, kappa, cm


def compute_confusion(cats_a, cats_b):
    """返回 confusion counter"""
    confusion = Counter()
    for a, b in zip(cats_a, cats_b):
        confusion[(a, b)] += 1
    return confusion


async def run_once(client, pairs_data, semaphore, run_id):
    """跑一次完整分类"""
    print(f"\n--- Run {run_id} 开始 ---")
    tasks = [
        process_pair(client, p["query"], p["cited_doc"]["content"], p["uncited_doc"]["content"], semaphore)
        for p in pairs_data
    ]
    results = await asyncio.gather(*tasks)
    print(f"--- Run {run_id} 完成 ---")

    # 保存
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
    path = os.path.join(OUTPUT_DIR, f"run_{run_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  保存到: {path}")
    print(f"  分布: {dict(cat_counter.most_common())}")

    return results


async def main():
    load_dotenv()

    with open(RAW_PAIRS_PATH, "r") as f:
        pairs_data = json.load(f)
    with open(LLM_RESULTS_PATH, "r") as f:
        llm_data = json.load(f)

    n = len(pairs_data)
    print(f"总条数: {n}, 模型: {MODEL}, 并发: {CONCURRENCY}, 运行次数: {NUM_RUNS}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    semaphore = asyncio.Semaphore(CONCURRENCY)

    # 跑三次
    all_results = []
    for run_id in range(1, NUM_RUNS + 1):
        results = await run_once(client, pairs_data, semaphore, run_id)
        all_results.append(results)

    # ── LLM taxonomy 的分类（合并 Window Truncation） ──
    llm_cats = []
    for r in llm_data["detailed_results"]:
        cat = r["category"]
        if cat == "Window Truncation":
            cat = "Systemic Exclusion"
        llm_cats.append(cat)

    # ── 每次 run 的分类 ──
    run_cats = []
    for results in all_results:
        run_cats.append([r["category"] for r in results])

    # ── 对比报告 ──
    print(f"\n{'='*70}")
    print(f"GPT-5 x3 Runs Comparison Report")
    print(f"{'='*70}")

    # 1. 分布对比
    print(f"\n--- Distribution ---")
    header = f"{'Category':<25} {'LLM':>8}"
    for i in range(NUM_RUNS):
        header += f" {'Run'+str(i+1):>8}"
    print(header)
    for cat in CATS:
        row = f"{cat:<25} {llm_cats.count(cat):>8}"
        for i in range(NUM_RUNS):
            row += f" {run_cats[i].count(cat):>8}"
        print(row)

    # 2. 每次 vs LLM 的一致性
    print(f"\n--- Each Run vs LLM Taxonomy ---")
    for i in range(NUM_RUNS):
        po, kappa, cm = compute_kappa(llm_cats, run_cats[i])
        print(f"  Run {i+1} vs LLM: Agreement={po*100:.1f}%, Kappa={kappa:.3f}")

    # 3. 三次之间的互相一致性
    print(f"\n--- Inter-Run Agreement ---")
    for i in range(NUM_RUNS):
        for j in range(i + 1, NUM_RUNS):
            po, kappa, cm = compute_kappa(run_cats[i], run_cats[j])
            print(f"  Run {i+1} vs Run {j+1}: Agreement={po*100:.1f}%, Kappa={kappa:.3f}")

    # 4. 三次全部一致的比例
    all_agree = sum(1 for k in range(n) if run_cats[0][k] == run_cats[1][k] == run_cats[2][k])
    print(f"\n--- 三次全部一致: {all_agree}/{n} ({all_agree/n*100:.1f}%) ---")

    # 5. 三次全部一致 且 与 LLM 一致
    all_agree_with_llm = sum(
        1 for k in range(n)
        if run_cats[0][k] == run_cats[1][k] == run_cats[2][k] == llm_cats[k]
    )
    print(f"--- 三次一致 + 与LLM一致: {all_agree_with_llm}/{n} ({all_agree_with_llm/n*100:.1f}%) ---")

    # 6. 三次全部一致 但 与 LLM 不一致
    all_agree_diff_llm = sum(
        1 for k in range(n)
        if run_cats[0][k] == run_cats[1][k] == run_cats[2][k] and run_cats[0][k] != llm_cats[k]
    )
    print(f"--- 三次一致 + 与LLM不一致: {all_agree_diff_llm}/{n} ({all_agree_diff_llm/n*100:.1f}%) ---")

    # 三次一致但与LLM不一致的 confusion
    print(f"\n--- 三次一致但与LLM不一致的分布 ---")
    stable_disagree = Counter()
    for k in range(n):
        if run_cats[0][k] == run_cats[1][k] == run_cats[2][k] and run_cats[0][k] != llm_cats[k]:
            stable_disagree[(llm_cats[k], run_cats[0][k])] += 1
    for (lc, ec), cnt in stable_disagree.most_common():
        print(f"  LLM={lc} → Evidence={ec}: {cnt}")

    # 7. Confusion matrix: 每次 run vs LLM
    for i in range(NUM_RUNS):
        print(f"\n--- Run {i+1} vs LLM Confusion Matrix ---")
        header = f"{'LLM \\\\ Run'+str(i+1):.<25}" + "".join(f"{c:>15}" for c in CATS)
        print(header)
        confusion = compute_confusion(llm_cats, run_cats[i])
        for lc in CATS:
            row = f"{lc:.<25}"
            for ec in CATS:
                row += f"{confusion.get((lc, ec), 0):>15}"
            print(row)

    # ── 保存综合报告 ──
    report = {
        "model": MODEL,
        "num_runs": NUM_RUNS,
        "total_pairs": n,
        "distribution": {
            "LLM": {cat: llm_cats.count(cat) for cat in CATS},
            **{f"Run_{i+1}": {cat: run_cats[i].count(cat) for cat in CATS} for i in range(NUM_RUNS)},
        },
        "vs_llm": {
            f"Run_{i+1}": {"agreement": compute_kappa(llm_cats, run_cats[i])[0], "kappa": compute_kappa(llm_cats, run_cats[i])[1]}
            for i in range(NUM_RUNS)
        },
        "inter_run": {
            f"Run_{i+1}_vs_Run_{j+1}": {"agreement": compute_kappa(run_cats[i], run_cats[j])[0], "kappa": compute_kappa(run_cats[i], run_cats[j])[1]}
            for i in range(NUM_RUNS) for j in range(i + 1, NUM_RUNS)
        },
        "all_three_agree": all_agree,
        "all_three_agree_with_llm": all_agree_with_llm,
        "all_three_agree_diff_llm": all_agree_diff_llm,
        "stable_disagree": {f"{lc} -> {ec}": cnt for (lc, ec), cnt in stable_disagree.most_common()},
    }
    report_path = os.path.join(OUTPUT_DIR, "comparison_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n综合报告已保存到: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())

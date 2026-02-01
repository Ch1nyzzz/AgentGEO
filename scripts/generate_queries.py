#!/usr/bin/env python3
"""
为 Parquet 文件生成 train/test 查询

使用 AgentGEO query_generator 模块生成高质量的 SEO 查询。
支持断点续传、并发处理、自动生成 doc_id。

使用方法:
    # 基本用法（使用默认配置）
    python scripts/generate_queries.py --input data/input.parquet

    # 指定训练和测试查询数量
    python scripts/generate_queries.py \
        --input data/input.parquet \
        --train-count 20 \
        --test-count 20

    # 自定义输出路径和并发数
    python scripts/generate_queries.py \
        --input data/input.parquet \
        --output data/output_with_queries.parquet \
        --concurrency 16

    # 测试模式（只处理前3个文档）
    python scripts/generate_queries.py --input data/input.parquet --limit 3

    # 重置 checkpoint 从头开始
    python scripts/generate_queries.py --input data/input.parquet --reset

必需字段:
    输入 Parquet 文件必须包含 'raw_html' 列
    可选字段: 'url', 'doc_id'
"""

import argparse
import hashlib
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

# 路径配置
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
QUERY_GEN_ROOT = REPO_ROOT / "query_generator"

# 添加到 Python 路径
for path in [str(REPO_ROOT), str(QUERY_GEN_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from query_generator import SEOQueryPipeline

# 默认配置
DEFAULT_TRAIN_COUNT = 20
DEFAULT_TEST_COUNT = 20
DEFAULT_RANDOM_SEED = 42
DEFAULT_CONCURRENCY = 8
DEFAULT_CONFIG_PATH = "query_generator/config.yaml"

# 线程安全锁
checkpoint_lock = threading.Lock()


def generate_doc_id(url: str, index: int) -> str:
    """根据 URL 或索引生成 doc_id"""
    if url:
        return hashlib.md5(url.encode()).hexdigest()[:12]
    return f"doc_{index:08d}"


def load_checkpoint(checkpoint_path: Path) -> dict:
    """加载 checkpoint"""
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"completed": [], "results": {}}


def save_checkpoint(checkpoint_path: Path, checkpoint: dict) -> None:
    """保存 checkpoint（线程安全）"""
    with checkpoint_lock:
        checkpoint["last_updated"] = datetime.now().isoformat()
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)


def process_single_document(
    doc_id: str,
    raw_html: str,
    train_count: int,
    test_count: int,
    config_path: str
) -> dict:
    """
    处理单个文档，生成查询

    Args:
        doc_id: 文档 ID
        raw_html: 原始 HTML 内容
        train_count: 训练集查询数量
        test_count: 测试集查询数量
        config_path: LLM 配置文件路径

    Returns:
        包含生成结果的字典
    """
    # 每个线程创建独立的 pipeline 实例
    pipeline = SEOQueryPipeline(config_path=config_path)

    # 使用 pipeline 处理文档
    result = pipeline.process(raw_html, file_uuid=doc_id)

    # 获取生成的查询
    train = result.dataset.train
    test = result.dataset.test

    # 采样到指定数量
    if len(train) > train_count:
        train = random.sample(train, train_count)
    if len(test) > test_count:
        test = random.sample(test, test_count)

    return {
        "doc_id": doc_id,
        "train_queries": train,
        "test_queries": test,
        "stats": {
            "pipeline_train": len(result.dataset.train),
            "pipeline_test": len(result.dataset.test),
            "sampled_train": len(train),
            "sampled_test": len(test),
        }
    }


def worker(task: dict) -> dict:
    """Worker 函数，处理单个任务"""
    try:
        result = process_single_document(
            task["doc_id"],
            task["raw_html"],
            task["train_count"],
            task["test_count"],
            task["config_path"]
        )
        result["success"] = True
        return result
    except Exception as e:
        return {
            "doc_id": task["doc_id"],
            "success": False,
            "error": str(e)
        }


def validate_dataframe(df: pd.DataFrame) -> None:
    """验证 DataFrame 是否包含必需字段"""
    if "raw_html" not in df.columns:
        raise ValueError("输入文件必须包含 'raw_html' 列")

    # 检查是否有空值
    null_count = df["raw_html"].isnull().sum()
    if null_count > 0:
        print(f"警告: 发现 {null_count} 个空的 raw_html 值")


def main():
    parser = argparse.ArgumentParser(
        description="为 Parquet 文件生成 train/test 查询",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  %(prog)s --input data/input.parquet

  # 自定义查询数量
  %(prog)s --input data/input.parquet --train-count 30 --test-count 10

  # 测试模式
  %(prog)s --input data/input.parquet --limit 3

  # 高并发处理
  %(prog)s --input data/input.parquet --concurrency 32
        """
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入 Parquet 文件路径（必须包含 'raw_html' 列）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 Parquet 文件路径（默认: {input}_with_queries.parquet）"
    )
    parser.add_argument(
        "--train-count",
        type=int,
        default=DEFAULT_TRAIN_COUNT,
        help=f"每个文档的训练查询数（默认: {DEFAULT_TRAIN_COUNT}）"
    )
    parser.add_argument(
        "--test-count",
        type=int,
        default=DEFAULT_TEST_COUNT,
        help=f"每个文档的测试查询数（默认: {DEFAULT_TEST_COUNT}）"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f"LLM 配置文件路径（默认: {DEFAULT_CONFIG_PATH}）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"随机种子（默认: {DEFAULT_RANDOM_SEED}）"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制处理的文档数（用于测试）"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="重置 checkpoint，从头开始"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"并发数（默认: {DEFAULT_CONCURRENCY}）"
    )

    args = parser.parse_args()

    # 设置随机种子
    random.seed(args.seed)

    # 确定输入/输出路径
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = REPO_ROOT / input_path

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = REPO_ROOT / output_path
    else:
        output_path = input_path.with_name(input_path.stem + "_with_queries.parquet")

    checkpoint_path = input_path.with_name(input_path.stem + "_query_checkpoint.json")

    # 确定配置文件路径
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    if not config_path.exists():
        print(f"警告: 配置文件不存在: {config_path}")
        print(f"将使用默认配置")

    # 检查输入文件
    if not input_path.exists():
        print(f"错误: 输入文件不存在: {input_path}")
        sys.exit(1)

    print("=" * 60)
    print("AgentGEO Query Generator")
    print("=" * 60)
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"配置文件: {config_path}")
    print("=" * 60)

    # 加载数据
    print(f"\n加载数据...")
    df = pd.read_parquet(input_path)
    print(f"总文档数: {len(df)}")

    # 验证数据
    try:
        validate_dataframe(df)
    except ValueError as e:
        print(f"错误: {e}")
        sys.exit(1)

    # 如果没有 doc_id 列，自动生成
    if "doc_id" not in df.columns:
        print("未找到 doc_id 列，自动生成...")
        df["doc_id"] = df.apply(
            lambda row: generate_doc_id(row.get("url", ""), row.name),
            axis=1
        )

    # 加载或重置 checkpoint
    if args.reset and checkpoint_path.exists():
        print("重置 checkpoint...")
        checkpoint_path.unlink()

    checkpoint = load_checkpoint(checkpoint_path)
    completed = set(checkpoint["completed"])
    print(f"已完成: {len(completed)} 个文档")

    # 确定要处理的文档
    docs_to_process = df if args.limit is None else df.head(args.limit)

    # 构建任务列表（跳过已完成的和空值）
    tasks = []
    skipped_null = 0
    for _, row in docs_to_process.iterrows():
        doc_id = row["doc_id"]
        raw_html = row["raw_html"]

        # 跳过空值
        if pd.isnull(raw_html) or not raw_html.strip():
            skipped_null += 1
            continue

        # 跳过已完成的
        if doc_id not in completed:
            tasks.append({
                "doc_id": doc_id,
                "raw_html": raw_html,
                "train_count": args.train_count,
                "test_count": args.test_count,
                "config_path": str(config_path),
            })

    total_tasks = len(tasks)
    print(f"\n待处理: {total_tasks} 个文档")
    if skipped_null > 0:
        print(f"跳过空值: {skipped_null} 个文档")
    print(f"并发数: {args.concurrency}")
    print(f"配置: train={args.train_count}, test={args.test_count}")
    print("=" * 60)

    if total_tasks == 0:
        print("\n所有文档已处理完成!")
    else:
        # 并发处理
        success_count = 0
        error_count = 0

        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            # 提交所有任务
            future_to_task = {executor.submit(worker, task): task for task in tasks}

            # 处理完成的任务
            for future in as_completed(future_to_task):
                result = future.result()
                doc_id = result["doc_id"]

                current_completed = len(checkpoint["completed"]) + 1
                progress = f"[{current_completed}/{len(completed) + total_tasks}]"

                if result["success"]:
                    # 更新 checkpoint（线程安全）
                    with checkpoint_lock:
                        checkpoint["results"][doc_id] = {
                            "train_queries": result["train_queries"],
                            "test_queries": result["test_queries"],
                        }
                        checkpoint["completed"].append(doc_id)

                    # 保存 checkpoint
                    save_checkpoint(checkpoint_path, checkpoint)

                    success_count += 1
                    stats = result.get("stats", {})
                    print(
                        f"{progress} ✓ {doc_id} | "
                        f"train={len(result['train_queries'])}/{stats.get('pipeline_train', '?')}, "
                        f"test={len(result['test_queries'])}/{stats.get('pipeline_test', '?')}"
                    )
                else:
                    error_count += 1
                    error_msg = result.get("error", "Unknown error")
                    print(f"{progress} ✗ {doc_id} | {error_msg}")

    # 合并结果到 DataFrame
    print("\n" + "=" * 60)
    print("合并结果到 DataFrame...")

    df["train_queries"] = df["doc_id"].map(
        lambda x: checkpoint["results"].get(x, {}).get("train_queries", [])
    )
    df["test_queries"] = df["doc_id"].map(
        lambda x: checkpoint["results"].get(x, {}).get("test_queries", [])
    )

    # 保存结果
    print(f"保存到: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    # 统计
    print("\n" + "=" * 60)
    print("完成!")
    print(f"  本次成功: {success_count if 'success_count' in locals() else 0}")
    print(f"  本次错误: {error_count if 'error_count' in locals() else 0}")
    print(f"  总完成: {len(checkpoint['completed'])}")
    print(f"  输出文件: {output_path}")
    print(f"  Checkpoint: {checkpoint_path}")

    # 验证输出
    print("\n验证输出...")
    df_out = pd.read_parquet(output_path)
    train_lens = df_out["train_queries"].apply(len)
    test_lens = df_out["test_queries"].apply(len)
    print(f"  train_queries 平均长度: {train_lens.mean():.1f}")
    print(f"  test_queries 平均长度: {test_lens.mean():.1f}")
    print(f"  有查询的文档数: {(train_lens > 0).sum()}/{len(df_out)}")
    print("=" * 60)


if __name__ == "__main__":
    main()

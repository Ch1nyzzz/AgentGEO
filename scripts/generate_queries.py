#!/usr/bin/env python3
"""
Generate train/test queries for Parquet files

Uses the AgentGEO query_generator module to generate high-quality SEO queries.
Supports checkpoint resumption, concurrent processing, and automatic doc_id generation.

Usage:
    # Basic usage (with default configuration)
    python scripts/generate_queries.py --input data/input.parquet

    # Specify train and test query counts
    python scripts/generate_queries.py \
        --input data/input.parquet \
        --train-count 20 \
        --test-count 20

    # Custom output path and concurrency
    python scripts/generate_queries.py \
        --input data/input.parquet \
        --output data/output_with_queries.parquet \
        --concurrency 16

    # Test mode (process only first 3 documents)
    python scripts/generate_queries.py --input data/input.parquet --limit 3

    # Reset checkpoint and start from beginning
    python scripts/generate_queries.py --input data/input.parquet --reset

Required fields:
    Input Parquet file must contain 'raw_html' column
    Optional fields: 'url', 'doc_id'
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

# Path configuration
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
QUERY_GEN_ROOT = REPO_ROOT / "query_generator"

# Add to Python path
for path in [str(REPO_ROOT), str(QUERY_GEN_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from query_generator import SEOQueryPipeline

# Default configuration
DEFAULT_TRAIN_COUNT = 20
DEFAULT_TEST_COUNT = 20
DEFAULT_RANDOM_SEED = 42
DEFAULT_CONCURRENCY = 8
DEFAULT_CONFIG_PATH = "query_generator/config.yaml"

# Thread-safe lock
checkpoint_lock = threading.Lock()


def generate_doc_id(url: str, index: int) -> str:
    """Generate doc_id based on URL or index"""
    if url:
        return hashlib.md5(url.encode()).hexdigest()[:12]
    return f"doc_{index:08d}"


def load_checkpoint(checkpoint_path: Path) -> dict:
    """Load checkpoint"""
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"completed": [], "results": {}}


def save_checkpoint(checkpoint_path: Path, checkpoint: dict) -> None:
    """Save checkpoint (thread-safe)"""
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
    Process a single document and generate queries

    Args:
        doc_id: Document ID
        raw_html: Raw HTML content
        train_count: Number of training queries
        test_count: Number of test queries
        config_path: LLM configuration file path

    Returns:
        Dictionary containing generation results
    """
    # Create independent pipeline instance for each thread
    pipeline = SEOQueryPipeline(config_path=config_path)

    # Process document using pipeline
    result = pipeline.process(raw_html, file_uuid=doc_id)

    # Get generated queries
    train = result.dataset.train
    test = result.dataset.test

    # Sample to specified count
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
    """Worker function to process a single task"""
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
    """Validate that DataFrame contains required fields"""
    if "raw_html" not in df.columns:
        raise ValueError("Input file must contain 'raw_html' column")

    # Check for null values
    null_count = df["raw_html"].isnull().sum()
    if null_count > 0:
        print(f"Warning: Found {null_count} null raw_html values")


def main():
    parser = argparse.ArgumentParser(
        description="Generate train/test queries for Parquet files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  %(prog)s --input data/input.parquet

  # Custom query counts
  %(prog)s --input data/input.parquet --train-count 30 --test-count 10

  # Test mode
  %(prog)s --input data/input.parquet --limit 3

  # High concurrency processing
  %(prog)s --input data/input.parquet --concurrency 32
        """
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input Parquet file path (must contain 'raw_html' column)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output Parquet file path (default: {input}_with_queries.parquet)"
    )
    parser.add_argument(
        "--train-count",
        type=int,
        default=DEFAULT_TRAIN_COUNT,
        help=f"Number of training queries per document (default: {DEFAULT_TRAIN_COUNT})"
    )
    parser.add_argument(
        "--test-count",
        type=int,
        default=DEFAULT_TEST_COUNT,
        help=f"Number of test queries per document (default: {DEFAULT_TEST_COUNT})"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f"LLM configuration file path (default: {DEFAULT_CONFIG_PATH})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Random seed (default: {DEFAULT_RANDOM_SEED})"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents to process (for testing)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset checkpoint and start from beginning"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Concurrency level (default: {DEFAULT_CONCURRENCY})"
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Determine input/output paths
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

    # Determine configuration file path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    if not config_path.exists():
        print(f"Warning: Configuration file not found: {config_path}")
        print(f"Will use default configuration")

    # Check input file
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    print("=" * 60)
    print("AgentGEO Query Generator")
    print("=" * 60)
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Config file: {config_path}")
    print("=" * 60)

    # Load data
    print(f"\nLoading data...")
    df = pd.read_parquet(input_path)
    print(f"Total documents: {len(df)}")

    # Validate data
    try:
        validate_dataframe(df)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Auto-generate doc_id if column doesn't exist
    if "doc_id" not in df.columns:
        print("doc_id column not found, auto-generating...")
        df["doc_id"] = df.apply(
            lambda row: generate_doc_id(row.get("url", ""), row.name),
            axis=1
        )

    # Load or reset checkpoint
    if args.reset and checkpoint_path.exists():
        print("Resetting checkpoint...")
        checkpoint_path.unlink()

    checkpoint = load_checkpoint(checkpoint_path)
    completed = set(checkpoint["completed"])
    print(f"Completed: {len(completed)} documents")

    # Determine documents to process
    docs_to_process = df if args.limit is None else df.head(args.limit)

    # Build task list (skip completed and null values)
    tasks = []
    skipped_null = 0
    for _, row in docs_to_process.iterrows():
        doc_id = row["doc_id"]
        raw_html = row["raw_html"]

        # Skip null values
        if pd.isnull(raw_html) or not raw_html.strip():
            skipped_null += 1
            continue

        # Skip completed documents
        if doc_id not in completed:
            tasks.append({
                "doc_id": doc_id,
                "raw_html": raw_html,
                "train_count": args.train_count,
                "test_count": args.test_count,
                "config_path": str(config_path),
            })

    total_tasks = len(tasks)
    print(f"\nPending: {total_tasks} documents")
    if skipped_null > 0:
        print(f"Skipped null values: {skipped_null} documents")
    print(f"Concurrency: {args.concurrency}")
    print(f"Config: train={args.train_count}, test={args.test_count}")
    print("=" * 60)

    if total_tasks == 0:
        print("\nAll documents have been processed!")
    else:
        # Concurrent processing
        success_count = 0
        error_count = 0

        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(worker, task): task for task in tasks}

            # Process completed tasks
            for future in as_completed(future_to_task):
                result = future.result()
                doc_id = result["doc_id"]

                current_completed = len(checkpoint["completed"]) + 1
                progress = f"[{current_completed}/{len(completed) + total_tasks}]"

                if result["success"]:
                    # Update checkpoint (thread-safe)
                    with checkpoint_lock:
                        checkpoint["results"][doc_id] = {
                            "train_queries": result["train_queries"],
                            "test_queries": result["test_queries"],
                        }
                        checkpoint["completed"].append(doc_id)

                    # Save checkpoint
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

    # Merge results into DataFrame
    print("\n" + "=" * 60)
    print("Merging results into DataFrame...")

    df["train_queries"] = df["doc_id"].map(
        lambda x: checkpoint["results"].get(x, {}).get("train_queries", [])
    )
    df["test_queries"] = df["doc_id"].map(
        lambda x: checkpoint["results"].get(x, {}).get("test_queries", [])
    )

    # Save results
    print(f"Saving to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    # Statistics
    print("\n" + "=" * 60)
    print("Complete!")
    print(f"  This run succeeded: {success_count if 'success_count' in locals() else 0}")
    print(f"  This run failed: {error_count if 'error_count' in locals() else 0}")
    print(f"  Total completed: {len(checkpoint['completed'])}")
    print(f"  Output file: {output_path}")
    print(f"  Checkpoint: {checkpoint_path}")

    # Validate output
    print("\nValidating output...")
    df_out = pd.read_parquet(output_path)
    train_lens = df_out["train_queries"].apply(len)
    test_lens = df_out["test_queries"].apply(len)
    print(f"  train_queries average length: {train_lens.mean():.1f}")
    print(f"  test_queries average length: {test_lens.mean():.1f}")
    print(f"  Documents with queries: {(train_lens > 0).sum()}/{len(df_out)}")
    print("=" * 60)


if __name__ == "__main__":
    main()

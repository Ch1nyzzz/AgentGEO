import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai_client import DEFAULT_MODEL, OpenAIClient


def load_competitor_urls(args: argparse.Namespace) -> list[str]:
    urls = list(args.competitor_url or [])
    if args.competitors_file:
        text = args.competitors_file.read_text(encoding="utf-8")
        if args.competitors_file.suffix == ".json":
            data = json.loads(text)
            urls.extend(data if isinstance(data, list) else data.get("competitor_urls", []))
        else:
            urls.extend(line.strip() for line in text.splitlines() if line.strip())
    return urls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score whether GPT cites a target URL.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--target-url", required=True)
    parser.add_argument(
        "--competitor-url",
        action="append",
        default=[],
        help="Competitor/source URL. Repeat for multiple competitors.",
    )
    parser.add_argument(
        "--competitors-file",
        type=Path,
        help="Text file with one competitor URL per line, or JSON list.",
    )
    parser.add_argument("--target-position", type=int)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-output-tokens", type=int, default=8000)
    parser.add_argument("--output-path", type=Path)
    return parser.parse_args()


async def main() -> None:
    load_dotenv()
    args = parse_args()
    result = await OpenAIClient().score(
        query=args.query,
        target_url=args.target_url,
        competitor_urls=load_competitor_urls(args),
        model=args.model,
        target_position=args.target_position,
        max_output_tokens=args.max_output_tokens,
    )

    payload = json.dumps(result, indent=2, ensure_ascii=False)
    print(payload)
    if args.output_path:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())

"""
Create embeddings via the OpenAI-compatible API.

- Reads a JSON file whose items contain `raw_text` and `rewrite_text`.
- Submits embedding requests (chunked) to OpenAI compatible endpoint.
- Writes embeddings back into the same JSON structure under `raw_embeddings` and `rewrite_embeddings`.

Usage example:
  python embedding_openai.py --input query_test_preprocessed.json --output query_test_embedded_openai.json
"""

import argparse
import json
import os
from typing import Dict, List, Sequence

from openai import OpenAI

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/gemini-embedding-001"
DEFAULT_API_KEY = ""
DEFAULT_CHUNK_SIZE = 64


def ensure_api_key(cli_key: str | None) -> str:
    key = (
        (cli_key or "").strip()
        or os.getenv("OPENROUTER_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
    )
    if not key:
        raise ValueError(
            "API key missing. Provide --api_key or set OPENROUTER_API_KEY / OPENAI_API_KEY."
        )
    return key


def load_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, got {type(data)}")
    return data


def save_json(path: str, data: Sequence[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(data), f, ensure_ascii=False, indent=2)


def embed_texts(
    texts: Sequence[str], *, client: OpenAI, model: str, chunk_size: int
) -> List[List[float]]:
    total = len(texts)
    results: List[List[float]] = [[] for _ in range(total)]
    size = chunk_size if chunk_size > 0 else max(1, total)

    for start in range(0, total, size):
        chunk = texts[start : start + size]
        if not chunk:
            continue
        try:
            resp = client.embeddings.create(model=model, input=chunk)
        except Exception as exc:  # pragma: no cover - network issues
            print(f"Error requesting embeddings for rows {start}-{start + len(chunk) - 1}: {exc}")
            continue

        for local_idx, item in enumerate(resp.data):
            try:
                global_idx = start + int(getattr(item, "index", local_idx))
            except Exception:
                global_idx = start + local_idx
            if 0 <= global_idx < total:
                try:
                    results[global_idx] = item.embedding  # type: ignore[attr-defined]
                except Exception:
                    results[global_idx] = []

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Openai embeddings via OpenAI-compatible API and write back to JSON file."
    )
    parser.add_argument("--input", help="Input JSON path.")
    parser.add_argument(
        "--output",
        help="Output JSON path. Default: overwrite input file.",
    )
    parser.add_argument(
        "--api_key",
        default=DEFAULT_API_KEY,
        help="Openai API key (fallback: OPENROUTER_API_KEY / OPENAI_API_KEY).",
    )
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL.")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="Embedding model name, e.g., openai/text-embedding-3-small."
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Number of rows per embedding request (<=0 for single request).",
    )
    parser.add_argument(
        "--stringify",
        action="store_true",
        help="If set, store embeddings as single-line JSON strings instead of numeric arrays.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = ensure_api_key(args.api_key)

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    data = load_json(args.input)
    base_name = os.path.splitext(os.path.basename(args.input))[0]

    raw_texts = [item.get("raw_text", "") for item in data]
    
    rewrite_texts = [str(item.get("rewrite_text", "")).split(">：")[-1] for item in data]

    print(f"Loaded {len(data)} rows from {args.input}")

    print("Submitting raw_text embedding requests ...")
    raw_embeddings = embed_texts(
        raw_texts,
        client=client,
        model=args.model,
        chunk_size=args.chunk_size,
    )
    print("Raw embeddings done.")

    print("Submitting rewrite_text embedding requests ...")
    rewrite_embeddings = embed_texts(
        rewrite_texts,
        client=client,
        model=args.model,
        chunk_size=args.chunk_size,
    )
    print("Rewrite embeddings done.")

    def maybe_stringify(vec: List[float]) -> str | List[float]:
        if args.stringify:
            # compact JSON for single-line storage
            return json.dumps(vec, ensure_ascii=False)
        return vec

    for idx, item in enumerate(data):
        item["rewrite_text"] = str(item["rewrite_text"]).split(">：")[-1]
        item["raw_embeddings"] = maybe_stringify(raw_embeddings[idx])
        item["rewrite_embeddings"] = maybe_stringify(rewrite_embeddings[idx])

    output_path = args.output or args.input
    save_json(output_path, data)
    print(f"Wrote embeddings to {output_path}")


if __name__ == "__main__":
    main()

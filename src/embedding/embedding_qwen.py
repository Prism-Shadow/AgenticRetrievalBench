"""
Create Qwen embeddings in batch mode using the OpenAI-compatible API.

- Reads a JSON file whose items contain `raw_text` and `rewrite_text`.
- Submits two batch embedding jobs (raw/rewrite) using the OpenAI Python SDK pointed at Qwen's compatibility endpoint.
- Writes embeddings back into the same JSON structure under `raw_embeddings` and `rewrite_embeddings`.

Simple usage example:
  python embedding_qwen.py --input query_test_rewrite.json  --output query_test_embedded.json

Notes:
- All interactions (file upload, batch creation, output download) use the OpenAI-compatible API only.
- The script keeps a local copy of batch outputs in `batch_files/` for inspection.
"""

import argparse
import json
import math
import os
import time
from typing import Dict, List, Sequence

from openai import OpenAI

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "text-embedding-v4"
DEFAULT_API_KEY = ""
DEFAULT_BATCH_DIR = "batch_files"
DEFAULT_COMPLETION_WINDOW = "24h"
DEFAULT_POLL_INTERVAL = 30
DEFAULT_CHUNK_SIZE = 5000


def ensure_api_key(cli_key: str | None) -> str:
    key = (
        (cli_key or "").strip()
        or os.getenv("QWEN_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
    )
    if not key:
        raise ValueError(
            "API key missing. Provide --api_key or set QWEN_API_KEY / OPENAI_API_KEY."
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


def dump_jsonl(lines: Sequence[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def build_embedding_requests(texts: Sequence[str], model: str, prefix: str, start_idx: int = 0) -> List[Dict]:
    requests = []
    for offset, text in enumerate(texts):
        idx = start_idx + offset
        requests.append(
            {
                "custom_id": f"{prefix}-{idx}",
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {"model": model, "input": text},
            }
        )
    return requests


def upload_request_file(openai_client: OpenAI, path: str):
    with open(path, "rb") as f:
        return openai_client.files.create(file=f, purpose="batch")


def create_batch(openai_client: OpenAI, file_id: str, *, completion_window: str) -> object:
    return openai_client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/embeddings",
        completion_window=completion_window,
    )


def _extract_attr(obj, key: str):
    if hasattr(obj, key):
        val = getattr(obj, key)
        if val:
            return val
    data = getattr(obj, "data", None)
    if isinstance(data, dict) and data.get(key):
        return data[key]
    if hasattr(obj, "model_dump"):
        dumped = obj.model_dump()
        if dumped.get(key):
            return dumped[key]
        if dumped.get("data", {}).get(key):
            return dumped["data"][key]
    if isinstance(obj, dict):
        if obj.get(key):
            return obj[key]
        if obj.get("data", {}).get(key):
            return obj["data"][key]
    return None


def poll_batch(openai_client: OpenAI, batch_id: str, *, poll_interval: int) -> object:
    while True:
        status = openai_client.batches.retrieve(batch_id)
        state = _extract_attr(status, "status")
        output_file_id = _extract_attr(status, "output_file_id")
        if state in {"failed", "cancelled", "expired"}:
            raise RuntimeError(f"Batch {batch_id} ended with status {state}")
        if state == "completed" and output_file_id:
            return status
        time.sleep(poll_interval)


def download_to_path(openai_client: OpenAI, file_id: str, target_path: str) -> None:
    os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
    openai_client.files.content(file_id).write_to_file(target_path)


def parse_embedding_output(output_path: str) -> Dict[str, List[float]]:
    mapping: Dict[str, List[float]] = {}
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = obj.get("custom_id") or obj.get("id")
            body = obj.get("response", {}).get("body", {})
            data = body.get("data") or []
            if data and isinstance(data, list):
                embedding = data[0].get("embedding")
                if embedding:
                    mapping[str(cid)] = embedding
                    continue
            error = obj.get("error") or body.get("error")
            if error:
                print(f"Warning: request {cid} returned error: {error}")
    return mapping


def run_embedding_batches(
    texts: Sequence[str],
    *,
    model: str,
    prefix: str,
    base_name: str,
    batch_dir: str,
    openai_client: OpenAI,
    poll_interval: int,
    completion_window: str,
    chunk_size: int,
) -> List[List[float]]:
    total = len(texts)
    results: List[List[float]] = [[] for _ in range(total)]
    num_parts = math.ceil(total / chunk_size) if chunk_size > 0 else 1

    for part in range(num_parts):
        start = part * chunk_size
        end = total if chunk_size <= 0 else min(total, (part + 1) * chunk_size)
        slice_texts = texts[start:end]
        req_lines = build_embedding_requests(slice_texts, model, prefix, start_idx=start)
        req_path = os.path.join(batch_dir, f"{base_name}_{prefix}_p{part}.jsonl")
        dump_jsonl(req_lines, req_path)

        uploaded = upload_request_file(openai_client, req_path)
        file_id = _extract_attr(uploaded, "id") or _extract_attr(uploaded, "file_id")
        if not file_id:
            raise RuntimeError(f"Failed to obtain uploaded file id for part {part}: {uploaded}")

        batch = create_batch(openai_client, str(file_id), completion_window=completion_window)
        batch_id = _extract_attr(batch, "id")
        if not batch_id:
            raise RuntimeError(f"Failed to create batch for part {part}: {batch}")
        print(f"Submitted batch {batch_id} for {prefix} part {part} (rows {start}-{end-1})")

        final_status = poll_batch(openai_client, str(batch_id), poll_interval=poll_interval)
        output_file_id = _extract_attr(final_status, "output_file_id")
        if not output_file_id:
            raise RuntimeError(f"Batch {batch_id} completed without output_file_id: {final_status}")

        out_path = os.path.join(batch_dir, f"{base_name}_{prefix}_p{part}_output.jsonl")
        download_to_path(openai_client, str(output_file_id), out_path)

        mapping = parse_embedding_output(out_path)
        for idx in range(start, end):
            key = f"{prefix}-{idx}"
            if key in mapping:
                results[idx] = mapping[key]
            else:
                print(f"Warning: missing embedding for {key}; leaving empty list")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Qwen embeddings via OpenAI-compatible Batch API and write back to JSON file."
    )
    parser.add_argument("--input", help="Input JSON path.")
    parser.add_argument("--output", help="Output JSON path. Default: overwrite input file.")
    parser.add_argument("--api_key", default=DEFAULT_API_KEY, help="Qwen API key (fallback: QWEN_API_KEY / GOOGLE_API_KEY / OPENAI_API_KEY).")
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Embedding model name, e.g., text-embedding-v4.")
    parser.add_argument("--batch_dir", default=DEFAULT_BATCH_DIR, help="Directory for temp batch files.")
    parser.add_argument("--poll_interval", type=int, default=DEFAULT_POLL_INTERVAL, help="Seconds between batch status checks.")
    parser.add_argument("--completion_window", default=DEFAULT_COMPLETION_WINDOW, help="Batch completion window, e.g., 24h or 72h.")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE, help="Number of rows per batch job (<=0 for single batch).")
    parser.add_argument("--stringify", action="store_true", help="If set, store embeddings as single-line JSON strings instead of numeric arrays.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = ensure_api_key(args.api_key)

    openai_client = OpenAI(api_key=api_key, base_url=args.base_url)

    data = load_json(args.input)
    base_name = os.path.splitext(os.path.basename(args.input))[0]

    raw_texts = [item.get("raw_text", "") for item in data]
    # split
    rewrite_texts = [item.get("rewrite_text", "").split(">：")[-1] for item in data]

    print(f"Loaded {len(data)} rows from {args.input}")

    print("Submitting raw_text embedding batches ...")
    raw_embeddings = run_embedding_batches(
        raw_texts,
        model=args.model,
        prefix="raw",
        base_name=base_name,
        batch_dir=args.batch_dir,
        openai_client=openai_client,
        poll_interval=args.poll_interval,
        completion_window=args.completion_window,
        chunk_size=args.chunk_size,
    )
    print("Raw embeddings done.")

    print("Submitting rewrite_text embedding batches ...")
    rewrite_embeddings = run_embedding_batches(
        rewrite_texts,
        model=args.model,
        prefix="rewrite",
        base_name=base_name,
        batch_dir=args.batch_dir,
        openai_client=openai_client,
        poll_interval=args.poll_interval,
        completion_window=args.completion_window,
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

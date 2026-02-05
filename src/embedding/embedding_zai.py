"""Run the embedding stage using the intermediate rewrite output with ZAI SDK.

This script consumes the JSON produced by `rewrite_zai.py`, submits batch
embedding jobs for `raw_text` (and `rewrite_text` when present), then assembles
the final `<data_basename>_embedded.json` file with the same structure as
the original monolithic pipeline.
"""

"""
usage:
python embedding_zai.py --input your_rewrites.json --output your_output.json
"""

import argparse
import json
import os
import time
from typing import Dict, Iterable, List, Sequence, Tuple

from zai import ZhipuAiClient

DEFAULT_API_KEY = ""
DEFAULT_EMBEDDING_MODEL = "embedding-3"


# IO helpers
def dump_json(data: Iterable[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(data), f, ensure_ascii=False, indent=2)


# batch helpers
def ensure_api_key(api_key: str | None) -> str:
    key = api_key or os.getenv("ZHIPU_API_KEY")
    if not key:
        raise ValueError("API key missing. Provide --api_key or set ZHIPU_API_KEY.")
    return key


def write_request(lines: Sequence[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def run_batch(
    client: ZhipuAiClient,
    request_file: str,
    *,
    endpoint: str,
    description: str,
    result_file: str,
    error_file: str | None = None,
    poll_interval: int = 20,
) -> str:
    with open(request_file, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint=endpoint,
        auto_delete_input_file=True,
        metadata={"description": description},
    )

    while True:
        status = client.batches.retrieve(batch.id)
        if status.status == "completed":
            break
        if status.status in {"failed", "expired", "cancelled"}:
            raise RuntimeError(f"Batch job ended with status: {status.status}")
        time.sleep(poll_interval)

    os.makedirs(os.path.dirname(result_file) or ".", exist_ok=True)
    client.files.content(status.output_file_id).write_to_file(result_file)
    if status.error_file_id and error_file:
        os.makedirs(os.path.dirname(error_file) or ".", exist_ok=True)
        client.files.content(status.error_file_id).write_to_file(error_file)
    return result_file


# embedding helpers
def build_embedding_body(model: str, text: str) -> Dict:
    return {"model": model, "input": text}


def prepare_embedding_requests(
    raws: Sequence[str], rewrites: Sequence[str], model: str, path: str
) -> None:
    lines = []
    for idx, text in enumerate(raws):
        lines.append(
            {
                "custom_id": f"embed-raw-{idx}",
                "method": "POST",
                "url": "/v4/embeddings",
                "body": build_embedding_body(model, text),
            }
        )
    for idx, text in enumerate(rewrites):
        lines.append(
            {
                "custom_id": f"embed-rewrite-{idx}",
                "method": "POST",
                "url": "/v4/embeddings",
                "body": build_embedding_body(model, text),
            }
        )
    write_request(lines, path)


def prepare_raw_only_embedding_requests(
    raws: Sequence[str], model: str, path: str
) -> None:
    lines = []
    for idx, text in enumerate(raws):
        lines.append(
            {
                "custom_id": f"embed-raw-{idx}",
                "method": "POST",
                "url": "/v4/embeddings",
                "body": build_embedding_body(model, text),
            }
        )
    write_request(lines, path)


def extract_embedding(obj: Dict) -> List[float]:
    body = obj.get("response", {}).get("body", {})
    data = body.get("data") or []
    if not data or not isinstance(data, list):
        raise ValueError("Missing data in embedding response")
    embedding = data[0].get("embedding")
    if isinstance(embedding, str):
        try:
            embedding = json.loads(embedding)
        except Exception:
            pass
    if not isinstance(embedding, list):
        raise ValueError("Embedding format not supported")
    return embedding  # type: ignore[return-value]


def parse_embedding_results(
    result_file: str, total: int, include_rewrite: bool = True
) -> Tuple[List[List[float]], List[List[float]]]:
    raw_map: Dict[int, List[float]] = {}
    rewrite_map: Dict[int, List[float]] = {}
    with open(result_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = obj.get("custom_id", "")
            try:
                emb = extract_embedding(obj)
            except Exception:
                continue
            if cid.startswith("embed-raw-"):
                idx = int(cid.replace("embed-raw-", ""))
                raw_map[idx] = emb
            elif cid.startswith("embed-rewrite-"):
                idx = int(cid.replace("embed-rewrite-", ""))
                rewrite_map[idx] = emb
    raw_embeddings = [raw_map.get(idx, []) for idx in range(total)]
    rewrite_embeddings = (
        [rewrite_map.get(idx, []) for idx in range(total)] if include_rewrite else raw_embeddings
    )
    return raw_embeddings, rewrite_embeddings


def load_rewrite_payload(path: str) -> Tuple[bool, str, List[Dict]]:
    """Load the rewrite output in either legacy or flat record format.

    Supported formats:
    1) Legacy (old rewrite_glm.py): [{"has_rewrite": bool, "source_data": str, "records": [...]}]
    2) Flat list (medical_subset_preprocessed.json style): [{"id": ..., "raw_text": ..., "rewrite_text": ...}, ...]
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    provided_has_rewrite: bool | None = None
    source_data = ""
    records: List[Dict] = []

    if isinstance(obj, list):
        if obj and isinstance(obj[0], dict) and "records" in obj[0]:
            payload = obj[0]
            provided_has_rewrite = payload.get("has_rewrite")
            source_data = payload.get("source_data", "")
            records = payload.get("records") or []
        else:
            records = obj  # already flat list of records
    elif isinstance(obj, dict) and "records" in obj:
        provided_has_rewrite = obj.get("has_rewrite")
        source_data = obj.get("source_data", "")
        records = obj.get("records") or []
    elif isinstance(obj, dict):
        records = [obj]

    if not records:
        raise ValueError("No records found in rewrite JSON")

    # Ensure rewrite_text exists for each record
    for rec in records:
        if "rewrite_text" not in rec:
            rec["rewrite_text"] = rec.get("raw_text", "")

    # Decide whether rewrite embeddings are needed
    if provided_has_rewrite is None:
        has_rewrite = any(rec.get("rewrite_text", "") != rec.get("raw_text", "") for rec in records)
    else:
        has_rewrite = bool(provided_has_rewrite)

    if not source_data:
        source_data = path

    return has_rewrite, source_data, records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate embeddings from rewrite output.")
    parser.add_argument("--input", required=True, help="Path to intermediate JSON produced by rewrite_zai.py.")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--output", help="Final output path. Default: <data_basename>_embedded.json")
    parser.add_argument("--api_key", default=DEFAULT_API_KEY, help="Zhipu API key (fallback to ZHIPU_API_KEY).")
    parser.add_argument("--batch_dir", default="batch_files", help="Where to place temporary batch files.")
    parser.add_argument("--poll_interval", type=int, default=20, help="Seconds between batch status checks.")
    parser.add_argument("--stringify", type=bool, default=True, help="Stringify the embeddings output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = ensure_api_key(args.api_key)
    client = ZhipuAiClient(api_key=api_key)

    print("步骤 1/4：加载改写中间结果 ...")
    has_rewrite, source_data, records = load_rewrite_payload(args.input)
    raw_texts = [r["raw_text"] for r in records]
    # split
    rewrite_texts = [r["rewrite_text"].split(">：")[-1] for r in records]
    base_name = os.path.splitext(os.path.basename(source_data or args.input))[0]
    output_path = args.output or f"{base_name}_embedded_glm.json"

    embed_request = os.path.join(args.batch_dir, f"{base_name}_embedding_request.jsonl")
    embed_result = os.path.join(args.batch_dir, f"{base_name}_embedding_result.jsonl")
    embed_error = os.path.join(args.batch_dir, f"{base_name}_embedding_error.jsonl")

    print("步骤 2/4：准备并提交向量批处理任务 ...")
    if has_rewrite:
        prepare_embedding_requests(raw_texts, rewrite_texts, args.model, embed_request)
    else:
        prepare_raw_only_embedding_requests(raw_texts, args.model, embed_request)
    run_batch(
        client,
        embed_request,
        endpoint="/v4/embeddings",
        description="embedding",
        result_file=embed_result,
        error_file=embed_error,
        poll_interval=args.poll_interval,
    )

    print("步骤 3/4：解析向量结果 ...")
    raw_embeddings, rewrite_embeddings = parse_embedding_results(
        embed_result, len(records), include_rewrite=has_rewrite
    )

    print("步骤 4/4：组装并写出最终结果 ...")
    rows = []
    for idx, rec in enumerate(records):
        rows.append(
            {
                "id": rec["id"],
                "raw_text": rec["raw_text"],
                "rewrite_text": rec["rewrite_text"].split(">：")[-1],
                # Store embeddings as single-line JSON strings for downstream consumers.
                "raw_embeddings": json.dumps(raw_embeddings[idx], ensure_ascii=False),
                "rewrite_embeddings": json.dumps(rewrite_embeddings[idx], ensure_ascii=False),
            }
        )
    dump_json(rows, output_path)
    print(f"全部完成：已写出 {len(rows)} 条记录到 {output_path}")


if __name__ == "__main__":
    main()

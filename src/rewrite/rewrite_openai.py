"""
Rewrite TSV data via OpenAI-compatible batch API.
Outputs JSON fields: id, raw_text, rewrite_text.
"""

"""
usage:
python rewrite_openai.py --input path/to/input.tsv --rewrite_model deepseek-ai/DeepSeek-V3 --mode query
"""

import argparse
import json
import math
import os
import time
from typing import Dict, Iterable, List, Sequence, Tuple

from openai import OpenAI

DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_API_KEY = ""
DEFAULT_REWRITE_MODEL = "deepseek-ai/DeepSeek-V3"

REWRITE_QUERY_SYSTEM_PROMPT = (
    """
    # 角色
    你是多领域的搜索增强助手，每次处理一行文本。

    # 工作流程
    1 关键词提取
    1.1 根据用户查询意图，从原文提炼1个核心关键词
    1.2 基于原文联想生成3-5个用户可能使用的其他搜索关键词

    2 提问生成
    2.1 以核心关键词为基础生成重写问句，不拆原有固定搭配，避免删减原文。
    2.2 以联想生成的补充关键词为主语，追加1–2个问题，询问和重写问句不同的内容。

    3 回答生成
    3.1 1–2句，基于原文核心关键词和用户意图，提供对应的信息或常识。

    # 注意事项
    - 输入可能是单个问题或多轮对话，多轮对话的输入格式为"<对话历史> xxx <最新问题> yyy"，其中yyy为最新问题。
    - 对于多轮问答对话输入，请保证核心关键词来自输入中的最新问题，并以最新问题作为重写目标。
    - 禁止编造具体数值、药名、剂量、法条号、平台名等细节。
    - 避免删除非平凡关键词，尽量只做扩充。
    - 输出需包含「思维链」与「最终输出」两段，使用<思维链>和<最终输出>标题分隔，思维链简要列推理步骤。
    - 最终输出不包含"原文"、"重写问句"等副标题。

    # 输出格式（单行纯文本）
    <思维链>：1 关键词提取 (原文关键词 补充关键词) 2 提问生成 3 回答生成 <最终输出>：提问：原文 重写问句 补充问句A 补充问句B 回答：回应重写问句+延申补充信息
    """
)

REWRITE_PASSAGE_SYSTEM_PROMPT = (
    """
    # 角色
    你是多领域的搜索增强助手，每次处理一段文本。

    # 工作流程
    1 原文分析
    1.1 判断原文文本性质和功能
    1.2 基于原文功能，判断原文作者意图

    2 关键词提取
    2.1 根据原文性质和原文作者意图，从原文提炼1个核心检索关键词
    2.2 基于原文核心关键词联想生成3-5个用户可能使用的其他搜索关键词

    3 提问生成
    3.1 从核心关键词反推怎样的用户意图能匹配到原文，基于1个核心关键词生成1个重写问句，突出稀有词和关键属性/情境，避免和原文重合度过高或过于详细。
    3.2 基于联想生成的补充关键词为主语，追加1–2个主语不同的延伸问法，询问和重写问句不同的内容。

    4 回答生成
    4.1 优先保留原文，基于稀有词补全相关信息，但禁止编造型号、数值、药名、剂量、法条号、平台名等具体信息。

    # 注意事项
    - 核心关键词指最符合原文意图，用户最可能检索的关键词，而非最能概括原文的关键词。
    - 禁止删减原文，只做扩充。
    - 禁止编造具体数值、药名、剂量、法条号、平台名等细节。
    - 输出需包含「思维链」与「最终输出」两段，使用<思维链>和<最终输出>标题分隔，思维链简要列推理步骤。
    - 最终输出不包含"原文"、"重写问句"等副标题。

    # 输出格式（单行纯文本）
    <思维链>：1 原文分析（文本性质/功能+作者意图） 2 关键词提取 (原文关键词 补充关键词) 3 提问生成 4 回答生成 <最终输出>：提问：重写问句 补充问句A 补充问句B 回答：原文+回应重写问句+延申补充信息
    """
)


def load_tsv(path: str) -> Tuple[List[Tuple[str, str]], int]:
    records: List[Tuple[str, str]] = []
    seen: set[Tuple[str, str]] = set()
    total_rows = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            total_rows += 1
            rec_id = parts[0].strip()
            raw_text = "\t".join(parts[1:]).strip()
            record = (rec_id, raw_text)
            if record not in seen:
                seen.add(record)
                records.append(record)
    return records, total_rows


def dump_json(data: Iterable[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(data), f, ensure_ascii=False, indent=2)


def ensure_api_key(cli_key: str | None) -> str:
    key = cli_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("API key missing. Provide --api_key or set OPENAI_API_KEY.")
    return key


def write_request(lines: Sequence[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def build_rewrite_body(model: str, text: str, system_prompt: str) -> Dict:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        "temperature": 0.3,
        # "temperature": 0.6
    }


def prepare_rewrite_requests(
    records: Sequence[Tuple[str, str]], model: str, path: str, system_prompt: str
) -> None:
    lines = []
    for idx, (_, raw_text) in enumerate(records):
        lines.append(
            {
                "custom_id": f"rewrite-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": build_rewrite_body(model, raw_text, system_prompt),
            }
        )
    write_request(lines, path)


def parse_rewrite_results(result_file: str, total: int) -> List[str]:
    mapping: Dict[str, str] = {}
    with open(result_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = obj.get("custom_id", "")
            body = obj.get("response", {}).get("body", {})
            choices = body.get("choices") or []
            if choices:
                content = choices[0].get("message", {}).get("content", "")
                mapping[cid] = content.strip()
    return [mapping.get(f"rewrite-{idx}", "") for idx in range(total)]


def run_batch(
    client: OpenAI,
    request_file: str,
    *,
    endpoint: str,
    description: str,
    result_file: str,
    error_file: str | None = None,
    poll_interval: int = 20,
    completion_window: str = "24h",
    extra_body: Dict | None = None,
) -> str:
    with open(request_file, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")

    input_file_id = getattr(file_obj, "id", None) or getattr(file_obj, "file_id", None)
    data = getattr(file_obj, "data", None)
    if not input_file_id and isinstance(data, dict):
        input_file_id = data.get("id") or data.get("file_id")
    if not input_file_id and hasattr(file_obj, "model_dump"):
        dumped = file_obj.model_dump()
        input_file_id = (
            dumped.get("id")
            or dumped.get("file_id")
            or dumped.get("data", {}).get("id")
            or dumped.get("data", {}).get("file_id")
        )
    if not input_file_id and isinstance(file_obj, dict):
        input_file_id = (
            file_obj.get("id")
            or file_obj.get("file_id")
            or file_obj.get("data", {}).get("id")
            or file_obj.get("data", {}).get("file_id")
        )
    if not input_file_id:
        raise RuntimeError(f"Failed to obtain uploaded file id from response: {file_obj}")
    input_file_id = str(input_file_id)
    print(f"Uploaded file id: {input_file_id}")

    batch = client.batches.create(
        input_file_id=input_file_id,
        endpoint=endpoint,
        completion_window=completion_window,
        metadata={"description": description},
        extra_body=extra_body or {},
    )

    def _extract_file_id(obj, key: str) -> str | None:
        if hasattr(obj, key):
            val = getattr(obj, key)
            if val:
                return str(val)
        data_inner = getattr(obj, "data", None)
        if isinstance(data_inner, dict) and data_inner.get(key):
            return str(data_inner[key])
        if hasattr(obj, "model_dump"):
            dumped_inner = obj.model_dump()
            if dumped_inner.get(key):
                return str(dumped_inner[key])
            if dumped_inner.get("data", {}).get(key):
                return str(dumped_inner["data"][key])
        if isinstance(obj, dict):
            if obj.get(key):
                return str(obj[key])
            if obj.get("data", {}).get(key):
                return str(obj["data"][key])
        return None

    while True:
        status = client.batches.retrieve(batch.id)
        state = getattr(status, "status", None) or (getattr(status, "data", {}) or {}).get("status")
        output_file_id = _extract_file_id(status, "output_file_id")
        error_file_id = _extract_file_id(status, "error_file_id")
        counts = getattr(status, "request_counts", None)

        if state in {"failed", "expired", "cancelled"}:
            raise RuntimeError(f"Batch job ended with status: {state}")

        if state == "completed" and output_file_id:
            print(
                f"Batch {batch.id} completed. output_file_id={output_file_id}, "
                f"error_file_id={error_file_id}, counts={counts}"
            )
            break

        if state in {"in_queue", "in_progress", "finalizing"} or (state == "completed" and not output_file_id):
            time.sleep(poll_interval)
            continue

        time.sleep(poll_interval)
        status_retry = client.batches.retrieve(batch.id)
        output_file_id = _extract_file_id(status_retry, "output_file_id")
        if output_file_id:
            status = status_retry
            break
        raise RuntimeError(f"Unexpected batch state '{state}', output_file_id not ready. Raw status: {status}")

    def _download_with_retry(file_id: str, target_path: str, max_retry: int = 5, wait: int = 5):
        last_err = None
        for attempt in range(1, max_retry + 1):
            try:
                if str(file_id).startswith("http"):
                    import urllib.request

                    with urllib.request.urlopen(file_id) as resp, open(target_path, "wb") as out:
                        out.write(resp.read())
                else:
                    client.files.content(file_id).write_to_file(target_path)
                return
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < max_retry:
                    time.sleep(wait)
                else:
                    raise last_err

    os.makedirs(os.path.dirname(result_file) or ".", exist_ok=True)
    _download_with_retry(output_file_id, result_file)
    if error_file_id and error_file:
        os.makedirs(os.path.dirname(error_file) or ".", exist_ok=True)
        _download_with_retry(error_file_id, error_file)
    return result_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use OpenAI-compatible batch API to rewrite text only."
    )
    parser.add_argument("--input", required=True, help="TSV file with id and raw text columns.")
    parser.add_argument(
        "--rewrite_model",
        default=DEFAULT_REWRITE_MODEL,
        help="Rewrite model name. If empty, skip rewrite and use raw_text as rewrite_text.",
    )
    parser.add_argument(
        "--mode",
        choices=["query", "passage"],
        default="query",
        help="Rewrite mode: query uses REWRITE_QUERY_SYSTEM_PROMPT, passage uses REWRITE_PASSAGE_SYSTEM_PROMPT.",
    )
    parser.add_argument("--output", help="Output json path. Default: <data_basename>_rewrite.json")
    parser.add_argument("--api_key", help="SiliconFlow/OpenAI compatible key.")
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL, help="SiliconFlow OpenAI-compatible endpoint.")
    parser.add_argument("--batch_dir", default="batch_files", help="Where to place temporary batch files.")
    parser.add_argument("--poll_interval", type=int, default=20, help="Seconds between batch status checks.")
    parser.add_argument("--completion_window", default="24h", help="Batch completion window, e.g., 24h or 72h.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = ensure_api_key(args.api_key)
    client = OpenAI(api_key=api_key, base_url=args.base_url)
    system_prompt = REWRITE_QUERY_SYSTEM_PROMPT if args.mode == "query" else REWRITE_PASSAGE_SYSTEM_PROMPT

    print("Step 1/2: loading TSV data ...")
    records, total_rows = load_tsv(args.input)
    if not records:
        raise ValueError(f"No valid rows found in {args.input}")
    if total_rows > len(records):
        print(
            f"Step 1/2 done: read {total_rows} rows, deduplicated to {len(records)} rows "
            f"(removed {total_rows - len(records)} duplicates)."
        )
    else:
        print(f"Step 1/2 done: read {len(records)} valid rows.")

    base = os.path.splitext(os.path.basename(args.input))[0]
    output_path = args.output or f"{base}_rewrite.json"

    has_rewrite = bool(args.rewrite_model and args.rewrite_model.strip())
    if has_rewrite:
        print("Step 2/2: preparing and submitting rewrite batches ...")
        chunk_size = 4500
        total = len(records)
        num_parts = math.ceil(total / chunk_size)
        rewrite_texts: List[str] = [""] * total
        for part_idx in range(num_parts):
            start = part_idx * chunk_size
            end = min(total, (part_idx + 1) * chunk_size)
            part_records = records[start:end]
            req_path = os.path.join(args.batch_dir, f"{base}_rewrite_request_p{part_idx}.jsonl")
            res_path = os.path.join(args.batch_dir, f"{base}_rewrite_result_p{part_idx}.jsonl")
            print(f"  Batch {part_idx + 1}/{num_parts}: rows {start}–{end - 1}")
            prepare_rewrite_requests(part_records, args.rewrite_model, req_path, system_prompt)
            run_batch(
                client,
                req_path,
                endpoint="/v1/chat/completions",
                description=f"rewrite_part_{part_idx}",
                result_file=res_path,
                poll_interval=args.poll_interval,
                completion_window=args.completion_window,
            )
            parsed = parse_rewrite_results(res_path, len(part_records))
            for offset, text in enumerate(parsed):
                rewrite_texts[start + offset] = text
        print("Step 2/2 done: rewrite results parsed and merged.")
    else:
        print("Step 2/2: skipping rewrite, using raw_text as rewrite_text ...")
        rewrite_texts = [r[1] for r in records]
        print("Step 2/2 done: reuse raw_text.")

    rows = []
    for idx, (rid, raw_text) in enumerate(records):
        rows.append(
            {
                "id": rid,
                "raw_text": raw_text,
                "rewrite_text": rewrite_texts[idx],
            }
        )
    dump_json(rows, output_path)
    print(f"All done: wrote {len(rows)} records to {output_path}")


if __name__ == "__main__":
    main()

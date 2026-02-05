"""Run the rewrite stage via ZhipuAI SDK.

Given a TSV file (id \t raw_text), produce an intermediate JSON file that
contains the raw text and the model rewrite. This file is later consumed by
`embedding_zai.py` to generate embeddings and assemble the final
`*_rewrite.json` output.
"""

"""
usage:
python rewrite_zai.py --input your_query/your_passage --rewrite_model glm-4-plus --mode query/passage
"""

import argparse
import json
import os
import time
from typing import Dict, Iterable, List, Sequence, Tuple

from zai import ZhipuAiClient

DEFAULT_API_KEY = ""
DEFAULT_REWRITE_MODEL = "glm-4-plus"

REWRITE_QUERY_SYSTEM_PROMPT = (
    """
    # 角色
    你是多领域的搜索增强助手，每次处理一行文本。

    # 工作流程
    1 关键词提取
    1.1 从原文提炼核心关键词，最终输出时仅保留一个
    1.2 基于原文联想生成3-5个用户可能使用的其他搜索关键词

    2 提问生成
    2.1 以核心关键词为基础生成重写问句，不拆原有固定搭配，避免删减原文。
    2.2 以联想生成的补充关键词为主语，追加1–2个问题，询问和重写问句不同的内容。

    3 回答生成
    3.1 1–2句，基于原文核心关键词和用户意图，提供对应的信息或常识。

    # 注意事项
    - 输入可能是单个问题或多轮对话，多轮对话的输入格式为"<对话历史> xxx </对话历史> yyy"，其中yyy为最新问题。
    - 对于多轮问答对话输入，请保证核心关键词来自输入中的最新问题，并以最新问题作为重写目标。
    - 禁止编造具体数值、药名、剂量、法条号、平台名等细节。
    - 避免删除非平凡关键词，尽量只做扩充。
    - 输出需包含「思维链」与「最终输出」两段，使用<思维链>和<最终输出>标题分隔，思维链简要列推理步骤。

    # 输出格式（单行纯文本）
    <思维链>：1 关键词提取 (原文关键词 补充关键词) 2 提问生成 3 回答生成 <最终输出>：提问：重写问句 补充问句A 补充问句B 回答：非平凡关键词回应+延申回答
    """
)

REWRITE_PASSAGE_SYSTEM_PROMPT = (
    """
    # 角色
    你是多领域的搜索增强助手，每次处理一段文本。

    # 工作流程
    1 关键词提取
    1.1 从原文提炼核心关键词
    1.2 基于原文核心关键词联想生成3-5个用户可能使用的其他搜索关键词

    2 提问生成
    2.1 从原文反推怎样的用户意图能匹配到原文，基于核心关键词生成一个重写问句，突出稀有词和关键属性/情境，避免和原文重合度过高或过于详细。
    2.2 基于联想生成的补充关键词为主语，追加1–2个主语不同的延伸问法，询问和重写问句不同的内容。

    3 回答生成
    3.1 优先保留原文，基于稀有词补全相关信息，但禁止编造型号、数值、药名、剂量、法条号、平台名等具体信息。

    # 注意事项
    - 禁止编造具体数值、药名、剂量、法条号、平台名等细节。
    - 避免删除非平凡关键词，尽量只做扩充。
    - 输出需包含「思维链」与「最终输出」两段，使用<思维链>和<最终输出>标题分隔，思维链简要列推理步骤。

    # 输出格式（单行纯文本）
    <思维链>：1 关键词提取 (原文关键词 补充关键词) 2 提问生成 3 回答生成 <最终输出>：提问：重写问句 补充问句A 补充问句B 回答：非平凡关键词回应+延申回答
    """
)


# IO helpers
def load_tsv(path: str) -> Tuple[List[Tuple[str, str]], int]:
    """Load TSV with id and raw text columns; deduplicate identical rows."""
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


# rewrite helpers
def build_rewrite_body(model: str, text: str, system_prompt: str) -> Dict:
    """Keep the payload compact for easy extension."""
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        "temperature": 0.3,
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
                "url": "/v4/chat/completions",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite raw text via ZhipuAI batch API.")
    parser.add_argument("--input", required=True, help="TSV file with id and raw text columns.")
    parser.add_argument("--rewrite_model", default=DEFAULT_REWRITE_MODEL, help="Rewrite model name. If empty, skip rewrite and use raw_text as rewrite_text.")
    parser.add_argument(
        "--mode",
        choices=["query", "passage"],
        default="query",
        help="Rewrite mode: query uses REWRITE_QUERY_SYSTEM_PROMPT, passage uses REWRITE_PASSAGE_SYSTEM_PROMPT.",
    )
    parser.add_argument("--output", help="Intermediate JSON path. Default: <data_basename>_rewrites.json")
    parser.add_argument("--api_key", default=DEFAULT_API_KEY, help="Zhipu AI key (fallback to ZHIPU_API_KEY).")
    parser.add_argument("--batch_dir", default="batch_files", help="Where to place temporary batch files.")
    parser.add_argument("--poll_interval", type=int, default=20, help="Seconds between batch status checks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = ensure_api_key(args.api_key)
    client = ZhipuAiClient(api_key=api_key)
    system_prompt = REWRITE_QUERY_SYSTEM_PROMPT if args.mode == "query" else REWRITE_PASSAGE_SYSTEM_PROMPT

    print("步骤 1/3：加载 TSV 数据 ...")
    records, total_rows = load_tsv(args.input)
    if not records:
        raise ValueError(f"No valid rows found in {args.input}")
    if total_rows > len(records):
        print(f"步骤 1/3 完成：共读取 {total_rows} 行，去重后保留 {len(records)} 行（去除 {total_rows - len(records)} 行重复数据）。")
    else:
        print(f"步骤 1/3 完成：共读取 {len(records)} 行有效数据。")

    base = os.path.splitext(os.path.basename(args.input))[0]
    output_path = args.output or f"{base}_rewrites.json"

    rewrite_request = os.path.join(args.batch_dir, f"{base}_rewrite_request.jsonl")
    rewrite_result = os.path.join(args.batch_dir, f"{base}_rewrite_result.jsonl")

    has_rewrite = bool(args.rewrite_model and args.rewrite_model.strip())
    if has_rewrite:
        print("步骤 2/3：准备并提交改写批处理任务 ...")
        prepare_rewrite_requests(records, args.rewrite_model, rewrite_request, system_prompt)
        run_batch(
            client,
            rewrite_request,
            endpoint="/v4/chat/completions",
            description="rewrite",
            result_file=rewrite_result,
            poll_interval=args.poll_interval,
        )
        rewrite_texts = parse_rewrite_results(rewrite_result, len(records))
        print("步骤 2/3 完成：改写结果已解析。")
    else:
        print("步骤 2/3：跳过改写，直接使用 raw_text 作为 rewrite_text ...")
        rewrite_texts = [r[1] for r in records]
        print("步骤 2/3 完成：已使用原始文本。")

    print("步骤 3/3：写出改写中间结果 ...")
    rows = []
    for idx, (rid, raw) in enumerate(records):
        rows.append(
            {
                "id": rid,
                "raw_text": raw,
                "rewrite_text": rewrite_texts[idx],
                "raw_embeddings": [],
                "rewrite_embeddings": [],
            }
        )
    dump_json(rows, output_path)
    print(f"全部完成：已写出 {len(rows)} 条记录到 {output_path}")


if __name__ == "__main__":
    main()

"""Evaluate MongoDB BM25 search for raw and rewrite queries."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import jieba
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError


Query = Tuple[str, str, str]  # (id, raw_text, rewrite_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MongoDB BM25 results for raw_text and rewrite_text queries."
    )
    parser.add_argument(
        "--query-file",
        default="",
        help="Path to query_preprocessed.json (contains id/raw_text/rewrite_text).",
    )
    parser.add_argument(
        "--qrels-file",
        default="",
        help="TSV with qrels: qid <tab> unused <tab> doc_id <tab> relevance.",
    )
    parser.add_argument(
        "--mongo-uri",
        default="",
        help="MongoDB connection URI (overridden by env MONGO_URI if set).",
    )
    parser.add_argument(
        "--db-name",
        default="",
        help="MongoDB database name.",
    )
    parser.add_argument(
        "--raw-collection",
        default="raw_text",
        help="Collection containing raw passages (field: text).",
    )
    parser.add_argument(
        "--rewrite-collection",
        default="rewrite_text",
        help="Collection containing rewrite passages (field: text).",
    )
    parser.add_argument(
        "--raw-index",
        default="bm25_raw_text",
        help="Atlas Search index name for the raw collection (omit to use default index).",
    )
    parser.add_argument(
        "--rewrite-index",
        default="bm25_rewrite_text",
        help="Atlas Search index name for the rewrite collection (omit to use default index).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of hits to retrieve for each query (should be >=10).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate only the first N queries (for quick smoke tests).",
    )
    return parser.parse_args()


def load_queries(path: Path) -> List[Query]:
    data = json.loads(path.read_text(encoding="utf-8"))
    queries: List[Query] = []
    for item in data:
        qid = str(item.get("id") or "").strip()
        if not qid:
            continue
        raw = item.get("raw_text") or ""
        rewrite = item.get("rewrite_text") or ""
        queries.append((qid, raw, rewrite))
    return queries


def load_qrels(path: Path) -> Dict[str, Set[str]]:
    mapping: Dict[str, Set[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        # dev.tsv 格式约定为：qid\t(无意义)\tpid\t(无意义)
        if len(parts) < 3:
            continue
        qid = parts[0].strip()
        doc_id = parts[2].strip()
        if not qid or not doc_id:
            continue
        # 忽略第 2、4 列内容，只将 (qid, pid) 视为相关对
        mapping.setdefault(qid, set()).add(doc_id)
    return mapping


_warned_fallback = False


def search_top_k(collection: Collection, query_text: str, top_k: int, index_name: str | None) -> List[str]:
    if not query_text:
        return []

    # Jieba segmentation improves Chinese BM25 matching
    segmented = " ".join(jieba.lcut(query_text))

    # Prefer Atlas Search BM25
    search_stage = {"$search": {"text": {"query": segmented, "path": "text"}}}
    if index_name:
        search_stage["$search"]["index"] = index_name

    pipeline = [search_stage, {"$limit": top_k}, {"$project": {"_id": 1, "id": 1, "score": {"$meta": "searchScore"}}}]

    try:
        return [str(doc.get("_id") or doc.get("id")) for doc in collection.aggregate(pipeline)]
    except PyMongoError:
        # Fallback to classic text index for local MongoDB
        global _warned_fallback
        if not _warned_fallback:
            print("[WARN] $search not available; falling back to $text (ensure a text index on 'text').")
            _warned_fallback = True
        try:
            cursor = (
                collection.find({"$text": {"$search": segmented}}, {"score": {"$meta": "textScore"}, "id": 1})
                .sort([("score", {"$meta": "textScore"})])
                .limit(top_k)
            )
            return [str(doc.get("_id") or doc.get("id")) for doc in cursor]
        except PyMongoError as exc:
            print(f"[WARN] Text search failed: {exc}")
            return []


def evaluate(
    queries: Sequence[Query],
    qrels: Dict[str, Set[str]],
    collection: Collection,
    pick_text,
    top_k: int,
    index_name: str | None,
) -> Tuple[Dict[str, float], int, List[Dict[str, object]]]:
    top_k = max(top_k, 10)
    totals = {"mrr": 0.0, "rec1": 0.0, "rec5": 0.0, "rec10": 0.0}
    evaluated = 0
    sample_hits: List[Dict[str, object]] = []

    for qid, raw, rewrite in queries:
        relevant = qrels.get(qid)
        if not relevant:
            continue

        query_text = pick_text(raw, rewrite)
        hits = search_top_k(collection, query_text, top_k, index_name)
        evaluated += 1

        rank = None
        for idx, doc_id in enumerate(hits):
            if doc_id in relevant:
                rank = idx + 1
                break

        if rank is not None:
            if rank <= 10:
                totals["mrr"] += 1.0 / rank
            if rank == 1:
                totals["rec1"] += 1
            if rank <= 5:
                totals["rec5"] += 1
            if rank <= 10:
                totals["rec10"] += 1

        if len(sample_hits) < 5:
            sample_hits.append(
                {
                    "qid": qid,
                    "query": query_text,
                    "hits": hits[:5],
                    "relevant": sorted(relevant),
                }
            )

    if evaluated == 0:
        raise ValueError("No queries matched qrels; nothing to evaluate.")

    for key in totals:
        totals[key] /= evaluated
    return totals, evaluated, sample_hits


def print_report(label: str, metrics: Dict[str, float], count: int, top_k: int) -> None:
    print(f"\n{label} (queries: {count}, top_k: {top_k})")
    print(f"  MRR@10   : {metrics['mrr']:.4f}")
    print(f"  Recall@1 : {metrics['rec1']:.4f}")
    print(f"  Recall@5 : {metrics['rec5']:.4f}")
    print(f"  Recall@10: {metrics['rec10']:.4f}")


def _shorten(text: str, limit: int = 120) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def print_samples(label: str, samples: List[Dict[str, object]]) -> None:
    if not samples:
        print(f"\n{label}: no sample hits to show.")
        return

    print(f"\n{label} (top5 for first {len(samples)} queries; * = relevant)")
    for idx, sample in enumerate(samples, 1):
        hits = sample.get("hits") or []
        relevant = set(sample.get("relevant") or [])
        annotated_hits = [f"{doc_id}{' *' if doc_id in relevant else ''}" for doc_id in hits]
        query_text = sample.get("query") or ""
        print(f"  {idx}. qid={sample.get('qid')} | query=\"{_shorten(query_text)}\"")
        print(f"     top5: {', '.join(annotated_hits) if annotated_hits else '-'}")


def main() -> None:
    args = parse_args()

    query_path = Path(args.query_file)
    qrels_path = Path(args.qrels_file)

    if not query_path.is_file():
        raise FileNotFoundError(f"query file not found: {query_path}")
    if not qrels_path.is_file():
        raise FileNotFoundError(f"qrels file not found: {qrels_path}")

    queries = load_queries(query_path)
    if args.limit:
        queries = queries[: args.limit]

    qrels = load_qrels(qrels_path)

    mongo_uri = os.environ.get("MONGO_URI", args.mongo_uri)
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")

    db = client[args.db_name]
    raw_coll = db[args.raw_collection]
    rewrite_coll = db[args.rewrite_collection]

    raw_metrics, raw_count, raw_samples = evaluate(
        queries,
        qrels,
        raw_coll,
        lambda raw, rewrite: raw,
        args.top_k,
        args.raw_index,
    )
    rewrite_metrics, rewrite_count, rewrite_samples = evaluate(
        queries,
        qrels,
        rewrite_coll,
        lambda raw, rewrite: rewrite,
        args.top_k,
        args.rewrite_index,
    )

    top_k = max(args.top_k, 10)
    print_report(f"Raw query with collection {args.raw_collection}", raw_metrics, raw_count, top_k)
    print_report(f"Rewrite query with collection {args.rewrite_collection}", rewrite_metrics, rewrite_count, top_k)
    print_samples("Raw text samples", raw_samples)
    print_samples("Rewrite text samples", rewrite_samples)


if __name__ == "__main__":
    main()

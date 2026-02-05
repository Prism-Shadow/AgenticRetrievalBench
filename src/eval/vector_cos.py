"""Evaluate MongoDB vector (cosine) search for raw and rewrite queries."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

Query = Tuple[str, List[float], List[float]]  # (id, raw_embedding, rewrite_embedding)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MongoDB vector search results for raw and rewrite queries."
    )
    parser.add_argument(
        "--query-file",
        default="",
        help="Path to query_preprocessed.json (contains id/raw_text/rewrite_text and embeddings).",
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
        default="retrieval_benchmark",
        help="MongoDB database name.",
    )
    parser.add_argument(
        "--raw-collection",
        default="raw_text",
        help="Collection containing raw passages (field: embedding).",
    )
    parser.add_argument(
        "--rewrite-collection",
        default="rewrite_text",
        help="Collection containing rewrite passages (field: embedding).",
    )
    parser.add_argument(
        "--raw-index",
        default="vec_raw_embedding",
        help="Atlas Search knn index name for the raw collection (omit to use default index).",
    )
    parser.add_argument(
        "--rewrite-index",
        default="vec_rewrite_embedding",
        help="Atlas Search knn index name for the rewrite collection (omit to use default index).",
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


def parse_embedding(value) -> Optional[List[float]]:
    """Normalize embeddings that may arrive as a JSON string or list."""
    if value is None:
        return None
    if isinstance(value, list):
        try:
            return [float(v) for v in value]
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [float(v) for v in parsed]
        except json.JSONDecodeError:
            return None
    return None


def load_queries(path: Path) -> List[Query]:
    data = json.loads(path.read_text(encoding="utf-8"))
    queries: List[Query] = []
    for item in data:
        qid = str(item.get("id") or "").strip()
        if not qid:
            continue
        raw_emb = parse_embedding(item.get("raw_embeddings") or item.get("raw_embedding") or item.get("embedding"))
        rewrite_emb = parse_embedding(item.get("rewrite_embeddings") or item.get("rewrite_embedding") or item.get("embedding"))
        if raw_emb is None and rewrite_emb is None:
            continue
        queries.append((qid, raw_emb or [], rewrite_emb or []))
    return queries


def load_qrels(path: Path) -> Dict[str, Sequence[str]]:
    mapping: Dict[str, List[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        qid = parts[0].strip()
        doc_id = parts[2].strip()
        if not qid or not doc_id:
            continue
        mapping.setdefault(qid, []).append(doc_id)
    return mapping


_warned = False


def search_top_k(collection: Collection, embedding: List[float], top_k: int, index_name: str | None) -> List[str]:
    if not embedding:
        return []

    search_spec = {"knnBeta": {"vector": embedding, "path": "embedding", "k": top_k}}
    if index_name:
        search_spec["index"] = index_name

    pipeline = [
        {"$search": search_spec},
        {"$limit": top_k},
        {"$project": {"_id": 1, "id": 1, "score": {"$meta": "searchScore"}}},
    ]

    try:
        return [str(doc.get("_id") or doc.get("id")) for doc in collection.aggregate(pipeline)]
    except PyMongoError as exc:
        global _warned
        if not _warned:
            print(f"[WARN] Vector $search failed (is Atlas Search enabled?): {exc}")
            _warned = True
        return []


def evaluate(
    queries: Sequence[Query],
    qrels: Dict[str, Sequence[str]],
    collection: Collection,
    pick_embedding,
    top_k: int,
    index_name: str | None,
) -> Tuple[Dict[str, float], int, List[Dict[str, object]]]:
    top_k = max(top_k, 10)
    totals = {"mrr": 0.0, "rec1": 0.0, "rec5": 0.0, "rec10": 0.0}
    evaluated = 0
    sample_hits: List[Dict[str, object]] = []

    for qid, raw_emb, rewrite_emb in queries:
        relevant = qrels.get(qid)
        if not relevant:
            continue

        embedding = pick_embedding(raw_emb, rewrite_emb)
        hits = search_top_k(collection, embedding, top_k, index_name)
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
                    "dim": len(embedding),
                    "hits": hits[:5],
                    "relevant": list(relevant),
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


def print_samples(label: str, samples: List[Dict[str, object]]) -> None:
    if not samples:
        print(f"\n{label}: no sample hits to show.")
        return

    print(f"\n{label} (top5 for first {len(samples)} queries; * = relevant)")
    for idx, sample in enumerate(samples, 1):
        hits = sample.get("hits") or []
        relevant = set(sample.get("relevant") or [])
        annotated_hits = [f"{doc_id}{' *' if doc_id in relevant else ''}" for doc_id in hits]
        print(f"  {idx}. qid={sample.get('qid')} | embedding-dim={sample.get('dim')}")
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
        lambda raw_emb, rewrite_emb: raw_emb,
        args.top_k,
        args.raw_index,
    )
    rewrite_metrics, rewrite_count, rewrite_samples = evaluate(
        queries,
        qrels,
        rewrite_coll,
        lambda raw_emb, rewrite_emb: rewrite_emb,
        args.top_k,
        args.rewrite_index,
    )

    top_k = max(args.top_k, 10)
    print_report(f"Raw query with collection {args.raw_collection}", raw_metrics, raw_count, top_k)
    print_report(f"Rewrite query with collection {args.rewrite_collection}", rewrite_metrics, rewrite_count, top_k)
    print_samples("Raw embedding samples", raw_samples)
    print_samples("Rewrite embedding samples", rewrite_samples)


if __name__ == "__main__":
    main()

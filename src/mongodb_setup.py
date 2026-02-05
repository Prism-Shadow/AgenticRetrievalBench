"""
Utility script to load a preprocessed passages JSON file into two MongoDB collections:
 - raw_text      -> documents with fields { _id, text, embedding }
 - rewrite_text  -> documents with fields { _id, text, embedding }

Input files should look like `passage_preprocessed.json` with keys:
  - id
  - raw_text, rewrite_text
  - raw_embeddings, rewrite_embeddings (stringified JSON arrays or lists)

Usage example:
  python mongodb_setup.py \
      --input preprocessedData/medical/passage_preprocessed.json \
      --mongo-uri "mongodb+srv://user:pwd@cluster.mongodb.net" \
      --db-name your_db_name \
      --raw-collection raw_text \
      --rewrite-collection rewrite_text \
      --batch-size 500
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import jieba
from pymongo import MongoClient, TEXT, errors

from mongodb_config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DB_NAME,
    DEFAULT_INPUT,
    DEFAULT_KEEP_EXISTING,
    DEFAULT_MONGO_URI,
    DEFAULT_RAW_COLLECTION,
    DEFAULT_REWRITE_COLLECTION,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load passages into MongoDB collections.")
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Path to passage_preprocessed.json (can be absolute or relative).",
    )
    parser.add_argument(
        "--mongo-uri",
        default=DEFAULT_MONGO_URI,
        help="MongoDB connection URI. Can also be provided via env MONGO_URI.",
    )
    parser.add_argument(
        "--db-name",
        default=DEFAULT_DB_NAME,
        help="Target MongoDB database name.",
    )
    parser.add_argument(
        "--raw-collection",
        default=DEFAULT_RAW_COLLECTION,
        help="Collection name for raw_text documents.",
    )
    parser.add_argument(
        "--rewrite-collection",
        default=DEFAULT_REWRITE_COLLECTION,
        help="Collection name for rewrite_text documents.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of documents to insert per batch.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        default=DEFAULT_KEEP_EXISTING,
        help="Do not drop collections if they already exist (default: overwrite).",
    )
    return parser.parse_args()


def drop_existing_collection(db, collection_name: str, search_index_names: Sequence[str]) -> None:
    """Drop Atlas Search indexes, other indexes, and the collection if present."""
    if collection_name not in db.list_collection_names():
        return

    coll = db[collection_name]

    # Atlas Search indexes
    for idx_name in search_index_names:
        try:
            db.command("dropSearchIndex", collection_name, name=idx_name)
            print(f"Dropped Search Index {idx_name}")
        except Exception as exc:  # pragma: no cover
            # Non-fatal; continue even if Atlas Search isn't enabled
            print(
                f"Warning: failed to drop Atlas Search index '{idx_name}' on {collection_name}: {exc}"
            )

    # All secondary indexes (MongoDB keeps the _id index)
    try:
        coll.drop_indexes()
    except Exception as exc:  # pragma: no cover
        print(f"Warning: failed to drop indexes on {collection_name}: {exc}")

    coll.drop()
    print(f"Dropped collection {collection_name}")


def iter_json_array(path: Path, chunk_size: int = 1_000_000) -> Iterator[dict]:
    """
    Stream-parse a large JSON array file without loading it fully into memory.
    Compatible with files shaped like: [ { ... }, { ... }, ... ]
    """
    decoder = json.JSONDecoder()
    buffer = ""
    with path.open("r", encoding="utf-8") as f:
        # Prime buffer with any leading characters
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            buffer += chunk
            start = 0

            # Skip whitespace and opening bracket
            while start < len(buffer) and buffer[start] in " \r\n\t[":
                start += 1
            buffer = buffer[start:]
            start = 0

            while True:
                try:
                    obj, end = decoder.raw_decode(buffer, start)
                except json.JSONDecodeError:
                    # Need more data; break to read next chunk
                    break

                yield obj
                start = end

                # Skip comma and standard between objects
                while start < len(buffer) and buffer[start] in " \r\n\t,":
                    start += 1

            buffer = buffer[start:]


def parse_embedding(value) -> Optional[List[float]]:
    """Normalize embeddings that may arrive as a JSON string or list."""
    if value is None:
        return None
    if isinstance(value, list):
        return [float(v) for v in value]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [float(v) for v in parsed]
        except json.JSONDecodeError:
            pass
    return None


def chunked(seq: Iterable[dict], size: int) -> Iterator[List[dict]]:
    batch: List[dict] = []
    for item in seq:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def infer_embedding_dim(documents: Iterable[dict]) -> Optional[int]:
    """Return the first non-empty embedding length found in documents."""
    for doc in documents:
        embedding = doc.get("embedding")
        if embedding:
            return len(embedding)
    return None


def build_docs(record: dict) -> Tuple[Optional[dict], Optional[dict]]:
    raw_doc = None
    rewrite_doc = None

    def tokenize(text: str) -> str:
        # Pre-segment with jieba to improve Chinese BM25 recall
        return " ".join(jieba.lcut(text or ""))

    if "raw_text" in record:
        raw_text = record.get("raw_text", "")
        raw_doc = {
            "_id": record.get("id"),
            # Store jieba-segmented text for indexing/search; keep original for inspection
            "text": tokenize(raw_text),
            "origin": raw_text,
            "embedding": parse_embedding(
                record.get("raw_embeddings") or record.get("raw_embedding") or record.get("embedding")
            ),
        }

    if "rewrite_text" in record:
        rewrite_text = record.get("rewrite_text", "")
        rewrite_doc = {
            "_id": record.get("id"),
            "text": tokenize(rewrite_text),
            "origin": rewrite_text,
            "embedding": parse_embedding(
                record.get("rewrite_embeddings") or record.get("rewrite_embedding") or record.get("embedding")
            ),
        }

    return raw_doc, rewrite_doc


def insert_batches(collection, documents: Iterable[dict], batch_size: int) -> int:
    inserted = 0
    for batch in chunked(documents, batch_size):
        if not batch:
            continue
        try:
            result = collection.insert_many(batch, ordered=False)
            inserted += len(result.inserted_ids)
        except errors.BulkWriteError as exc:
            # Continue past duplicate key errors; count successful inserts
            inserted += exc.details.get("nInserted", 0)
    return inserted


def ensure_bm25_search_index(collection, index_name: str) -> None:
    """
    Create a BM25-capable Atlas Search index (lucene.standard) on the collection's ``text`` field.
    Also installs a classic text index as a local fallback when $search is unavailable.
    """
    definition = {
        "mappings": {
            "dynamic": False,
            "fields": {
                "text": {
                    "type": "string",
                    "analyzer": "lucene.standard",
                    "searchAnalyzer": "lucene.standard",
                }
            },
        }
    }

    try:
        collection.database.command(
            "createSearchIndexes",
            collection.name,
            indexes=[{"name": index_name, "definition": definition}],
        )
        print(f"Created Search Index {index_name}")
    except Exception as exc:  # pragma: no cover
        # Non-fatal: continue if Atlas Search isn't available
        print(f"Warning: failed to create Atlas Search index '{index_name}' on {collection.name}: {exc}")

    # Text index fallback for self-hosted MongoDB
    try:
        collection.create_index([("text", TEXT)], name=f"{index_name}_text_fallback")
    except Exception as exc:  # pragma: no cover
        print(f"Warning: failed to create text index fallback on {collection.name}: {exc}")


def ensure_vector_search_index(collection, index_name: str, dimensions: Optional[int]) -> None:
    """
    Create a vector (k-NN) Atlas Search index on the collection's ``embedding`` field.
    Skips creation when dimensions cannot be inferred.
    """
    if not dimensions:
        print(f"Skip creating vector index '{index_name}' for {collection.name}: embedding dimension unknown")
        return

    definition = {
        "mappings": {
            "dynamic": False,
            "fields": {
                "embedding": {
                    "type": "knnVector",
                    "dimensions": dimensions,
                    "similarity": "cosine",
                }
            },
        }
    }

    try:
        collection.database.command(
            "createSearchIndexes",
            collection.name,
            indexes=[{"name": index_name, "definition": definition}],
        )
        print(f"Created vector Search Index {index_name} (dims={dimensions})")
    except Exception as exc:  # pragma: no cover
        print(f"Warning: failed to create vector Search Index '{index_name}' on {collection.name}: {exc}")


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not args.mongo_uri:
        raise ValueError("Mongo URI not provided. Use --mongo-uri or set MONGO_URI environment variable.")

    client = MongoClient(args.mongo_uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")  # fail fast on connection issues

    db = client[args.db_name]

    # Overwrite existing collections unless user opts to keep them
    raw_search_index = "bm25_raw_text"
    rewrite_search_index = "bm25_rewrite_text"
    raw_vector_index = "vec_raw_embedding"
    rewrite_vector_index = "vec_rewrite_embedding"

    if not args.keep_existing:
        drop_existing_collection(
            db,
            args.raw_collection,
            [raw_search_index, raw_vector_index],
        )
        drop_existing_collection(
            db,
            args.rewrite_collection,
            [rewrite_search_index, rewrite_vector_index],
        )

    raw_coll = db[args.raw_collection]
    rewrite_coll = db[args.rewrite_collection]

    raw_coll.create_index("_id")
    rewrite_coll.create_index("_id")

    # BM25 (jieba tokens + standard analyzer) indexes on raw_text/rewrite_text collections
    ensure_bm25_search_index(raw_coll, index_name=raw_search_index)
    ensure_bm25_search_index(rewrite_coll, index_name=rewrite_search_index)

    raw_docs: List[dict] = []
    rewrite_docs: List[dict] = []

    for record in iter_json_array(input_path):
        raw_doc, rewrite_doc = build_docs(record)
        if raw_doc:
            raw_docs.append(raw_doc)
        if rewrite_doc:
            rewrite_docs.append(rewrite_doc)

    # Vector search indexes on embedding fields (dimensions inferred from data)
    raw_dim = infer_embedding_dim(raw_docs)
    rewrite_dim = infer_embedding_dim(rewrite_docs)
    ensure_vector_search_index(raw_coll, index_name=raw_vector_index, dimensions=raw_dim)
    ensure_vector_search_index(rewrite_coll, index_name=rewrite_vector_index, dimensions=rewrite_dim)

    raw_inserted = insert_batches(raw_coll, raw_docs, args.batch_size)
    rewrite_inserted = insert_batches(rewrite_coll, rewrite_docs, args.batch_size)

    print(
        f"Completed. Inserted {raw_inserted} documents into '{args.raw_collection}' "
        f"and {rewrite_inserted} into '{args.rewrite_collection}'."
    )


if __name__ == "__main__":
    main()

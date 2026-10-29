from __future__ import annotations

import time
from typing import Dict, List

import numpy as np
import redis
from redis.commands.search.field import NumericField, TagField, TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.exceptions import ResponseError

from src.config import Settings


def _to_str(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if value is None:
        return ""
    return str(value)


def get_redis_client(settings: Settings) -> redis.Redis:
    try:
        client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=False,
        )
        client.ping()
        return client
    except Exception as exc:
        raise RuntimeError("Cannot connect to Redis. Please run: docker compose up -d") from exc


def create_index(redis_client: redis.Redis, settings: Settings) -> None:
    schema = (
        TextField("text"),
        TagField("source"),
        NumericField("chunk_no"),
        NumericField("created_at"),
        VectorField(
            "embedding",
            "HNSW",
            {
                "TYPE": "FLOAT32",
                "DIM": settings.embedding_dim,
                "DISTANCE_METRIC": "COSINE",
            },
        ),
    )

    definition = IndexDefinition(prefix=[settings.redis_doc_prefix], index_type=IndexType.HASH)

    redis_client.ft(settings.redis_index_name).create_index(
        fields=schema,
        definition=definition,
    )


def save_chunk(
    redis_client: redis.Redis,
    settings: Settings,
    text: str,
    source: str,
    chunk_no: int,
    embedding: np.ndarray,
) -> str:
    key = f"{settings.redis_doc_prefix}{source}:{chunk_no}"
    mapping = {
        "text": text,
        "source": source.replace(" ", "_"),
        "chunk_no": chunk_no,
        "created_at": int(time.time()),
        "embedding": embedding.astype(np.float32).tobytes(),
    }
    redis_client.hset(key, mapping=mapping)
    return key


def search_similar_chunks(
    redis_client: redis.Redis,
    settings: Settings,
    query_embedding: np.ndarray,
    top_k: int,
) -> List[Dict[str, str]]:
    query = (
        Query(f"*=>[KNN {top_k} @embedding $vector AS score]")
        .sort_by("score")
        .return_fields("text", "source", "chunk_no", "score")
        .dialect(2)
    )

    params = {"vector": query_embedding.astype(np.float32).tobytes()}
    results = redis_client.ft(settings.redis_index_name).search(query, query_params=params)

    chunks: List[Dict[str, str]] = []
    for doc in results.docs:
        source = _to_str(getattr(doc, "source", ""))
        chunks.append(
            {
                "text": _to_str(getattr(doc, "text", "")),
                "source": source.replace("_", " "),
                "chunk_no": _to_str(getattr(doc, "chunk_no", "0")),
                "score": _to_str(getattr(doc, "score", "0")),
            }
        )
    return chunks


def clear_project_data(redis_client: redis.Redis, settings: Settings) -> None:
    try:
        redis_client.ft(settings.redis_index_name).dropindex(delete_documents=True)
        print("Old index deleted.")
    except ResponseError:
        print("No existing index found. Creating a fresh index.")

    cursor = 0
    while True:
        cursor, keys = redis_client.scan(cursor=cursor, match=f"{settings.redis_doc_prefix}*")
        if keys:
            redis_client.delete(*keys)
        if cursor == 0:
            break

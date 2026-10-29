from __future__ import annotations

from src.chunker import chunk_text
from src.config import load_settings
from src.document_loader import load_documents
from src.llm_client import LLMClient
from src.redis_store import (
    clear_project_data,
    create_index,
    get_redis_client,
    save_chunk,
)


def main() -> None:
    settings = load_settings()
    documents = load_documents(settings.data_dir)
    if not documents:
        raise RuntimeError(
            "No supported documents found in data folder. Please add .txt, .md, or .pdf files."
        )

    redis_client = get_redis_client(settings)
    llm_client = LLMClient(
        api_key=settings.gemini_api_key,
        embedding_model=settings.gemini_embedding_model,
        chat_model=settings.gemini_chat_model,
        embedding_dim=settings.embedding_dim,
        llm_provider=settings.llm_provider,
        embedding_provider=settings.embedding_provider,
        deepseek_api_key=settings.deepseek_api_key,
        deepseek_base_url=settings.deepseek_base_url,
        deepseek_chat_model=settings.deepseek_chat_model,
        debug=settings.gemini_debug,
    )

    clear_project_data(redis_client, settings)
    create_index(redis_client, settings)
    print(f"Created Redis index: {settings.redis_index_name}\n")

    total_chunks = 0
    for document in documents:
        print(f"Document: {document['source']}")
        chunks = chunk_text(
            document["text"],
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        print(f"Chunks created: {len(chunks)}")

        for chunk_no, chunk in enumerate(chunks, start=1):
            embedding = llm_client.get_embedding(chunk)
            save_chunk(
                redis_client=redis_client,
                settings=settings,
                text=chunk,
                source=document["source"],
                chunk_no=chunk_no,
                embedding=embedding,
            )
            total_chunks += 1
            print(f"Saved chunk {chunk_no}")

        print("")

    print("Ingestion completed successfully.")
    print(f"Total chunks saved: {total_chunks}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(error)
        raise SystemExit(1) from error

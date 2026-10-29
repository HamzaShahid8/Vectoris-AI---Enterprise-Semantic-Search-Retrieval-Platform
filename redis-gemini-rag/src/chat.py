from __future__ import annotations

import re
from typing import List

from src.config import load_settings
from src.llm_client import LLMClient
from src.redis_store import get_redis_client, search_similar_chunks

PROMPT_TEMPLATE = """You are a strict RAG assistant.

Rules:
1. Answer only using the context below.
2. Do not use outside knowledge.
3. If the answer is not available in the context, say:
   "I don't know based on the provided documents."
4. Keep the answer simple and clear.
5. Mention source file name if useful.

Context:
{retrieved_context}

User Question:
{user_question}

Answer:
"""


def build_context(chunks: List[dict]) -> str:
    context_parts = []
    for item in chunks:
        context_parts.append(
            f"Source: {item['source']}\n"
            f"Chunk No: {item['chunk_no']}\n"
            f"Text: {item['text']}"
        )
    return "\n\n".join(context_parts)


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {token for token in tokens if len(token) > 2}


def fallback_answer_from_chunks(chunks: List[dict], user_question: str) -> str:
    if not chunks:
        return "I don't know based on the provided documents."

    question_tokens = _tokenize(user_question)
    best_chunk = chunks[0]
    best_score = -1

    for chunk in chunks:
        chunk_text = chunk.get("text", "")
        chunk_tokens = _tokenize(chunk_text)
        score = len(question_tokens & chunk_tokens)
        if score > best_score:
            best_score = score
            best_chunk = chunk

    if best_score <= 0:
        return "I don't know based on the provided documents."

    text = best_chunk.get("text", "").strip()
    source = best_chunk.get("source", "unknown")
    if not text:
        return "I don't know based on the provided documents."
    return f"{text}\n\n(Source: {source})"


def main() -> None:
    settings = load_settings()
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

    print("Redis + Gemini RAG Chatbot")
    print("Ask questions from your documents.")
    print("Type exit to stop.\n")

    while True:
        user_question = input("You: ").strip()
        if user_question.lower() == "exit":
            print("Goodbye.")
            break
        if not user_question:
            continue

        question_embedding = llm_client.get_embedding(
            user_question,
            task_type="retrieval_query",
        )
        chunks = search_similar_chunks(
            redis_client=redis_client,
            settings=settings,
            query_embedding=question_embedding,
            top_k=settings.top_k,
        )

        if not chunks:
            print("No relevant context found in Redis.\n")
            continue

        print("\nRetrieved Chunks:")
        for idx, chunk in enumerate(chunks, start=1):
            print(f"{idx}. {chunk['source']}")

        context_text = build_context(chunks)
        prompt = PROMPT_TEMPLATE.format(
            retrieved_context=context_text,
            user_question=user_question,
        )

        try:
            answer = llm_client.ask_llm(prompt)
            if not answer:
                answer = "I don't know based on the provided documents."
        except RuntimeError as error:
            print(f"\nLLM unavailable right now: {error}")
            print("Using retrieved context fallback.\n")
            answer = fallback_answer_from_chunks(chunks, user_question)

        print(f"\nBot: {answer}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(error)
        raise SystemExit(1) from error

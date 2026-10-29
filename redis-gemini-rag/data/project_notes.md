# RAG Project Notes

This project builds a Retrieval-Augmented Generation system using Redis and Gemini.

Redis is used as a vector database. It stores document chunks and their embeddings.

Gemini is used for two purposes:
1. Creating embeddings from text.
2. Generating final answers from retrieved context.

The system should answer only from the indexed private documents.
If the answer is not available in the documents, the assistant should say that it does not know based on the provided documents.

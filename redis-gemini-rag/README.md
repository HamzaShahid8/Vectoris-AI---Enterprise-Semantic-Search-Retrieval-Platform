# Redis + DeepSeek RAG (Python)

This project is a simple RAG chatbot.

RAG means Retrieval-Augmented Generation.

First, the system reads your private documents.
Then it breaks them into small chunks.
Then it converts each chunk into embeddings.
An embedding is a list of numbers that represents the meaning of text.
These embeddings are stored in Redis.
When you ask a question, the question is also converted into an embedding.
Redis compares the question embedding with document embeddings.
It finds the most relevant chunks.
Those chunks are sent to the LLM.
The LLM creates an answer using only those chunks.

## Project Goal

Build a beginner-friendly Retrieval-Augmented Generation (RAG) chatbot using:

- Python
- Redis Stack (vector database)
- DeepSeek chat model
- Optional Gemini embeddings (or local embeddings)

The bot answers questions using only your indexed private documents.

## Folder Structure

```text
redis-gemini-rag/
│
├── data/
│   ├── company_policy.txt
│   ├── employee_benefits.txt
│   ├── it_support.md
│   ├── meeting_rules.txt
│   └── project_notes.md
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── document_loader.py
│   ├── chunker.py
│   ├── llm_client.py
│   ├── redis_store.py
│   ├── ingest.py
│   └── chat.py
│
├── .env.example
├── .env
├── .gitignore
├── requirements.txt
├── docker-compose.yml
└── README.md
```

## How It Works

### Ingestion flow

1. Load documents from `data/`
2. Extract and clean text
3. Create overlapping chunks
4. Generate embeddings (local or Gemini)
5. Store chunks + metadata + vectors in Redis

### Chat flow

1. Take question from terminal
2. Generate question embedding
3. Search top-k relevant chunks in Redis
4. Build strict RAG prompt with retrieved context
5. Ask DeepSeek and print answer

### Backend-only usage

This project has no HTTP server, web framework, or API routes. Run ingestion with
`python -m src.ingest`, then start the interactive backend with `python -m src.chat`.
Questions and answers are entered in the terminal; there is no backend URL or
HTTP endpoint to call.

## Installation (Step-by-Step)

### Step 1: Clone or create project

```bash
mkdir redis-gemini-rag
cd redis-gemini-rag
```

### Step 2: Create virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Mac/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Start Redis Stack

```bash
docker compose up -d
```

Check container:

```bash
docker ps
```

If host port 6379 is already in use, set both `REDIS_PORT` and
`REDIS_HOST_PORT` to an available port (for example `6380`) in `.env`.
`REDIS_PORT` is the port the Python backend connects to; `REDIS_HOST_PORT`
controls the host-side Docker port mapping. If port 8001 is occupied, set
`REDIS_INSIGHT_HOST_PORT` to another available port (for example `8002`).

### Step 5: Setup environment variables

Windows:

```powershell
copy .env.example .env
```

Mac/Linux:

```bash
cp .env.example .env
```

Then add your DeepSeek API key in `.env`:

```env
DEEPSEEK_API_KEY=your_real_deepseek_api_key
```

### Step 6: Add documents

Put `.txt`, `.md`, or `.pdf` files into:

```text
data/
```

### Step 7: Run ingestion

```bash
python -m src.ingest
```

### Step 8: Start chatbot

```bash
python -m src.chat
```

## Commands

- Ingest documents: `python -m src.ingest`
- Start chat: `python -m src.chat`

## Environment Modes

Default project mode is DeepSeek + local embeddings (no Gemini key needed):

```env
EMBEDDING_PROVIDER=local
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_real_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_CHAT_MODEL=deepseek-chat
```

The local embedding provider uses deterministic hashed word vectors; it does not
download or load a pretrained embedding model.

Optional Gemini mode:

```env
EMBEDDING_PROVIDER=gemini
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_real_gemini_api_key
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
GEMINI_CHAT_MODEL=gemini-2.0-flash
```

## Prompt Rules (Used by Chatbot)

The assistant is forced to:

1. Answer only from retrieved context
2. Avoid outside knowledge
3. Return this exact fallback if answer is missing:
   `I don't know based on the provided documents.`

## Example Questions

1. What is the attendance requirement?
2. What should kitchen staff wear?
3. What is Redis used for in this project?
4. Who is the CEO of Google?

Expected unknown-answer behavior:

`I don't know based on the provided documents.`

## Error Messages

The project provides clear errors:

- Missing DeepSeek API key:
  `Missing DEEPSEEK_API_KEY. Please add it to your .env file.`
- Redis not running:
  `Cannot connect to Redis. Please run: docker compose up -d`
- No documents found:
  `No supported documents found in data folder. Please add .txt, .md, or .pdf files.`
- No context found during chat:
  `No relevant context found in Redis.`
- LLM API issue:
  `<Provider> API request failed. Please check your API key and internet connection.`

## Manual Testing Checklist

1. Redis container running:
   - Run `docker ps`
   - Check `redis-gemini-rag` is running
2. Ingestion works:
   - Run `python -m src.ingest`
   - Check chunks are generated and saved
3. Document-based answer:
   - Run `python -m src.chat`
   - Ask: `What is the attendance requirement?`
4. Unknown question:
   - Ask: `What is the capital of Japan?`
   - Expect: `I don't know based on the provided documents.`

## Security Notes

- `.env` is ignored in `.gitignore`
- API keys are read only from environment variables
- No credentials are hardcoded in source files

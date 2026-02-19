# agentic-ai-rag-langgraph-project 🧪

A growing collection of agentic AI workflows built with **LangGraph**, **LangChain**, and **Ollama** — all running 100% locally, no API keys required.

The first workflow is an **Agentic RAG** (Retrieval-Augmented Generation) system that retrieves, grades, and answers questions from your own documents.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Agentic RAG** | Multi-node LangGraph graph: query rewriting → retrieval → grading → generation |
| 🧠 **Conversation Memory** | Remembers the last 3 turns — resolves pronouns and follow-up questions |
| ✍️ **Query Rewriting** | Rewrites vague questions into precise retrieval queries before searching |
| 📄 **Document Summarizer** | Summarizes every document in your knowledge base on demand |
| 🔄 **Multi-Model Support** | Switch between any locally installed Ollama model at runtime |
| 💾 **Persistent Vector Store** | ChromaDB index is built once and reused across sessions |

---

## 🏗️ Architecture

```
User Question
     │
     ▼
[rewrite_query]   ← LLM rephrases question using conversation history
     │
     ▼
[retrieve]        ← Semantic search in ChromaDB (top-4 chunks)
     │
     ▼
[grade_documents] ← Keyword-overlap filter (instant, no LLM call)
     │
     ├── relevant ──→ [generate]  ← LLM synthesizes answer from context + history
     │
     └── not_relevant → [fallback] ← LLM answers from own knowledge
```

Built with:
- [LangGraph](https://github.com/langchain-ai/langgraph) — stateful multi-node agent orchestration
- [LangChain](https://github.com/langchain-ai/langchain) — document loading, splitting, prompts
- [Ollama](https://ollama.com) — local LLM and embedding inference
- [ChromaDB](https://www.trychroma.com) — local vector store
- [Rich](https://github.com/Textualize/rich) — terminal UI

---

## 🚀 Prerequisites

1. **Python 3.9+**
2. **[Ollama](https://ollama.com)** installed and running
3. Required Ollama models pulled:

```bash
ollama pull qwen3:4b          # default LLM
ollama pull nomic-embed-text  # embeddings (required)
```

---

## ⚙️ Setup

```bash
# 1. Clone the repo
git clone https://github.com/venkateshfax/agentic-ai-rag-langgraph-project.git
cd agentic-ai-rag-langgraph-project

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Copy environment config
cp .env.example .env
```

---

## 📂 Adding Documents

Drop any `.pdf`, `.txt`, or `.md` files into the `docs/` folder:

```
docs/
├── my-paper.pdf
├── notes.txt
└── research.md
```

The knowledge base is built automatically on first run. To re-index after adding new files:

```bash
# Inside the CLI, type:
rebuild
```

---

## ▶️ Usage

```bash
./venv/bin/python main.py
```

### CLI Commands

| Command | Description |
|---|---|
| `<any question>` | Query the knowledge base |
| `/summarize` | Summarize all loaded documents |
| `/models` | List available Ollama models |
| `/model <name>` | Switch to a different LLM (e.g. `/model deepseek-r1:8b`) |
| `rebuild` | Re-index all documents in `docs/` |
| `exit` / `quit` | Exit the CLI |

### Example Session

```
You: What is LangGraph?
  [Node: rewrite_query]  → "LangGraph definition key concepts"
  [Node: retrieve]       → 2 documents
  [Node: grade_documents]→ relevant (1/2 kept)
  [Node: generate]       → ...

You: How does it compare to LangChain?
  [Node: rewrite_query]  → "LangGraph vs LangChain comparison differences"  ← uses history!
  ...

You: /summarize
  ╭── sample.txt ──────────────────╮
  │ LangGraph is a library for...  │
  ╰────────────────────────────────╯

You: /model deepseek-r1:8b
  Switched to model: deepseek-r1:8b
```

---

## 🗂️ Project Structure

```
langgraph-lab/
├── main.py              # CLI entry point (Rich REPL)
├── src/
│   ├── agent.py         # LangGraph state graph, nodes, ModelConfig
│   ├── summarizer.py    # Document summarization module
│   └── vector_store.py  # ChromaDB document loading and indexing
├── docs/                # Your knowledge base (PDF, TXT, MD)
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🛣️ Roadmap

- [ ] LangSmith tracing integration for observability
- [ ] Multi-agent workflow (researcher + writer agents)
- [ ] Web search tool node (DuckDuckGo / Tavily)
- [ ] Streaming output support
- [ ] REST API layer (FastAPI)
- [ ] Evaluation harness for RAG accuracy

---

## 🤝 Contributing

Contributions welcome! If you want to add a new agentic workflow:

1. Fork the repo
2. Create a branch: `git checkout -b feature/my-workflow`
3. Add your workflow under `src/` or a new `workflows/` directory
4. Open a PR with a description of the workflow and how to run it

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

"""Document summarizer for the Agentic RAG system."""

import os
from collections import defaultdict
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


SUMMARIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a document summarizer. Given several excerpts from a document, "
     "produce a concise summary (3-5 sentences) covering the main topics. "
     "Be factual, clear, and brief. Do not use filler phrases."),
    ("human",
     "Document: {source}\n\nExcerpts:\n{excerpts}\n\nSummary:"),
])


def summarize_documents(
    vector_store: Chroma,
    model_name: str,
    max_chunks_per_doc: int = 8,
) -> List[Tuple[str, str]]:
    """
    Fetch all chunks from the vector store, group by source file,
    and summarize each using the LLM.

    Returns a list of (display_name, summary) tuples sorted alphabetically.
    Returns an empty list if the store has no documents.
    """
    raw = vector_store.get()
    texts = raw.get("documents", [])
    metadatas = raw.get("metadatas", [])

    if not texts:
        return []

    # Group chunks by source file (LangChain loaders set metadata["source"])
    grouped: defaultdict = defaultdict(list)
    for text, meta in zip(texts, metadatas):
        source = (meta or {}).get("source", "Unknown")
        grouped[source].append(text)

    llm = ChatOllama(model=model_name, temperature=0.1, num_predict=300)
    chain = SUMMARIZE_PROMPT | llm

    results = []
    for source in sorted(grouped.keys()):
        chunks = grouped[source][:max_chunks_per_doc]
        excerpts = "\n\n---\n\n".join(chunks)
        display_name = os.path.basename(source)

        result = chain.invoke({"source": display_name, "excerpts": excerpts})
        summary = result.content.strip()
        results.append((display_name, summary))

    return results

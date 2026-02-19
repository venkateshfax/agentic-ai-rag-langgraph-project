"""Vector store management for the Agentic RAG system."""

import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document


CHROMA_DB_DIR = "./chroma_db"
DOCS_DIR = "./docs"


def get_embeddings(model: str = "nomic-embed-text") -> OllamaEmbeddings:
    """Get Ollama embeddings model."""
    return OllamaEmbeddings(model=model)


def load_documents(docs_dir: str = DOCS_DIR) -> List[Document]:
    """Load documents from the docs directory."""
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        docs_path.mkdir(parents=True)

    documents = []

    # Load PDFs
    pdf_files = list(docs_path.glob("**/*.pdf"))
    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        documents.extend(loader.load())

    # Load text files
    txt_files = list(docs_path.glob("**/*.txt"))
    for txt_file in txt_files:
        loader = TextLoader(str(txt_file))
        documents.extend(loader.load())

    # Load markdown files
    md_files = list(docs_path.glob("**/*.md"))
    for md_file in md_files:
        loader = TextLoader(str(md_file))
        documents.extend(loader.load())

    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return splitter.split_documents(documents)


def build_vector_store(docs_dir: str = DOCS_DIR, persist_dir: str = CHROMA_DB_DIR) -> Chroma:
    """Build or load a Chroma vector store from documents."""
    embeddings = get_embeddings()

    # If persist dir exists, load existing store
    if Path(persist_dir).exists() and any(Path(persist_dir).iterdir()):
        print(f"Loading existing vector store from {persist_dir}...")
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    print(f"Loading documents from {docs_dir}...")
    documents = load_documents(docs_dir)

    if not documents:
        print("No documents found. Creating empty vector store.")
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    print(f"Loaded {len(documents)} document(s). Splitting into chunks...")
    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunks. Building vector store...")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    print("Vector store built successfully.")
    return vector_store


def add_documents_to_store(vector_store: Chroma, documents: List[Document]) -> None:
    """Add new documents to an existing vector store."""
    chunks = split_documents(documents)
    vector_store.add_documents(chunks)
    print(f"Added {len(chunks)} chunks to vector store.")

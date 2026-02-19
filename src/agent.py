"""Agentic RAG using LangGraph with Ollama."""

from dataclasses import dataclass
from typing import TypedDict, List, Annotated
import operator

from langchain.schema import Document
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, END


# ── Model Configuration ────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """Runtime-mutable LLM model configuration. Closed over in build_rag_graph()."""
    model_name: str = "qwen3:4b"


# ── State ──────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """State shared across all nodes in the graph."""
    question: str
    documents: List[Document]
    generation: str
    messages: Annotated[List[BaseMessage], operator.add]
    grade: str          # "relevant" | "not_relevant"
    iterations: int


# ── Node helpers ───────────────────────────────────────────────────────────────

def get_llm(model_name: str, temperature: float = 0.0, num_predict: int = 512) -> ChatOllama:
    return ChatOllama(model=model_name, temperature=temperature, num_predict=num_predict)


# ── Node: retrieve ─────────────────────────────────────────────────────────────

def retrieve(state: AgentState, vector_store: Chroma) -> AgentState:
    """Retrieve relevant documents from the vector store."""
    print("\n[Node: retrieve]")
    question = state["question"]
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    documents = retriever.invoke(question)
    print(f"  Retrieved {len(documents)} document(s)")
    return {**state, "documents": documents, "iterations": state.get("iterations", 0) + 1}


# ── Node: grade_documents ──────────────────────────────────────────────────────
# Fast keyword-overlap grading — no extra LLM call, runs in milliseconds.

def _keyword_overlap(question: str, text: str, threshold: float = 0.15) -> bool:
    """Return True if enough question keywords appear in the document text."""
    import re
    stopwords = {"what", "is", "the", "a", "an", "are", "how", "does", "do",
                 "why", "who", "which", "in", "of", "to", "and", "or", "for"}
    q_words = {w for w in re.findall(r"\w+", question.lower()) if w not in stopwords and len(w) > 2}
    if not q_words:
        return True  # nothing to filter on → keep
    t_words = set(re.findall(r"\w+", text.lower()))
    overlap = len(q_words & t_words) / len(q_words)
    return overlap >= threshold


def grade_documents(state: AgentState) -> AgentState:
    """Grade retrieved documents using fast keyword-overlap (no LLM call)."""
    print("\n[Node: grade_documents]")
    question = state["question"]
    documents = state["documents"]

    relevant_docs = [doc for doc in documents if _keyword_overlap(question, doc.page_content)]

    grade = "relevant" if relevant_docs else "not_relevant"
    print(f"  Grade: {grade} ({len(relevant_docs)}/{len(documents)} docs kept)")
    return {**state, "documents": relevant_docs, "grade": grade}


# ── Node: generate ─────────────────────────────────────────────────────────────

GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
     "If the context is insufficient, say so honestly. Be concise and accurate."),
    ("human",
     "Context:\n{context}\n\nQuestion: {question}"),
])


def generate(state: AgentState, model_config: ModelConfig) -> AgentState:
    """Generate an answer from the retrieved documents."""
    print("\n[Node: generate]")
    llm = get_llm(model_config.model_name, temperature=0.2)
    chain = GENERATE_PROMPT | llm

    question = state["question"]
    documents = state["documents"]
    context = "\n\n".join(doc.page_content for doc in documents)

    result = chain.invoke({"context": context, "question": question})
    generation = result.content.strip()
    print(f"  Generated answer ({len(generation)} chars)")

    messages = state.get("messages", []) + [
        HumanMessage(content=question),
        AIMessage(content=generation),
    ]
    return {**state, "generation": generation, "messages": messages}


# ── Node: fallback ─────────────────────────────────────────────────────────────

FALLBACK_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Answer the question using your own knowledge. "
     "Be concise. Note at the end that no relevant documents were found in the knowledge base."),
    ("human", "Question: {question}"),
])


def web_search_fallback(state: AgentState, model_config: ModelConfig) -> AgentState:
    """Fallback: answer from model knowledge when docs are not relevant."""
    print("\n[Node: web_search_fallback]")
    llm = get_llm(model_config.model_name, temperature=0.2)
    chain = FALLBACK_PROMPT | llm

    result = chain.invoke({"question": state["question"]})
    generation = result.content.strip()
    print("  Generated fallback answer")

    messages = state.get("messages", []) + [
        HumanMessage(content=state["question"]),
        AIMessage(content=generation),
    ]
    return {**state, "generation": generation, "messages": messages}


# ── Conditional edges ──────────────────────────────────────────────────────────

def decide_after_grade(state: AgentState) -> str:
    """Route after grading: generate if relevant, fallback otherwise."""
    if state["grade"] == "relevant":
        return "generate"
    return "fallback"


# ── Build graph ────────────────────────────────────────────────────────────────

def build_rag_graph(vector_store: Chroma, model_config: ModelConfig) -> StateGraph:
    """Build and compile the LangGraph RAG agent."""

    def _retrieve(state):
        return retrieve(state, vector_store)

    def _generate(state):
        return generate(state, model_config)

    def _fallback(state):
        return web_search_fallback(state, model_config)

    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", _retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", _generate)
    workflow.add_node("fallback", _fallback)

    workflow.set_entry_point("retrieve")

    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_after_grade,
        {"generate": "generate", "fallback": "fallback"},
    )
    workflow.add_edge("generate", END)
    workflow.add_edge("fallback", END)

    return workflow.compile()


def run_query(graph, question: str, model_config: ModelConfig = None) -> str:
    """Run a single query through the RAG graph."""
    initial_state: AgentState = {
        "question": question,
        "documents": [],
        "generation": "",
        "messages": [],
        "grade": "",
        "iterations": 0,
    }
    final_state = graph.invoke(initial_state)
    return final_state["generation"]

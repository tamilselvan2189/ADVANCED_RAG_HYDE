# app.py
"""
Entry point: compare three RAG strategies on a single user query.
  1. Baseline RAG (query → retriever)
  2. Query-Rewriting RAG (LLM-rewritten query → retriever)
  3. HyDE RAG (synthetic answer → embedding → retriever)
"""
from __future__ import annotations

import os
from typing import List, Tuple

from langchain_core.documents import Document

from utils.common import PROJECT_ROOT, get_llm, load_config
from utils.hyde_generator import generate_hyde_embedding
from utils.loader import load_and_chunk_docs
from utils.query_rewriter import rewrite_query
from utils.retriever import get_retriever


DOCS_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "insurance_docs")

ANSWER_SYSTEM_PROMPT = "Answer using ONLY the provided context. If unknown, say so."


# --------------------------------------------------------------------------
# Shared answer helper
# --------------------------------------------------------------------------
def _answer_from_docs(query: str, docs: List[Document], llm) -> str:
    context = "\n\n".join(d.page_content for d in docs)
    prompt = [
        ("system", ANSWER_SYSTEM_PROMPT),
        ("human", f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"),
    ]
    return llm.invoke(prompt).content.strip()


# --------------------------------------------------------------------------
# Three RAG strategies
# --------------------------------------------------------------------------
def run_baseline(query: str, retriever, llm) -> Tuple[str, List[Document]]:
    docs = retriever.invoke(query)
    return _answer_from_docs(query, docs, llm), docs


def run_rewriting(query: str, retriever, llm) -> Tuple[str, List[Document]]:
    rewritten = rewrite_query(query)
    print(f"\n🔁 Rewritten Query → {rewritten}")
    docs = retriever.invoke(rewritten)
    return _answer_from_docs(rewritten, docs, llm), docs


def run_hyde(query: str, retriever, llm) -> Tuple[str, List[Document]]:
    hyde_vector = generate_hyde_embedding(query)
    docs = retriever.vectorstore.similarity_search_by_vector(
        hyde_vector, k=retriever.search_kwargs.get("k", 5)
    )
    return _answer_from_docs(query, docs, llm), docs


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def _print_section(title: str) -> None:
    bar = "=" * max(len(title) + 4, 30)
    print(f"\n{bar}\n  {title}\n{bar}")


def main() -> None:
    config = load_config()
    llm = get_llm(config)
    provider = config["llm"]["provider"].upper()

    print(f"\n🚀 Active Provider: {provider}")

    chunks = load_and_chunk_docs(DOCS_DIR)
    if not chunks:
        raise RuntimeError(
            f"No chunks produced from {DOCS_DIR}. "
            "Verify the folder contains non-empty .txt or .pdf files."
        )

    retriever = get_retriever(chunks_if_needed=chunks, config=config)

    query = input("\n🔍 Enter your question: ").strip()
    if not query:
        print("Empty query — exiting.")
        return

    _print_section("✅ BASELINE RAG")
    baseline_ans, _ = run_baseline(query, retriever, llm)
    print(f"Q: {query}\nA: {baseline_ans}")

    _print_section("✅ QUERY REWRITING + RAG")
    rewriting_ans, _ = run_rewriting(query, retriever, llm)
    print(f"A: {rewriting_ans}")

    _print_section("✅ HyDE RAG")
    hyde_ans, _ = run_hyde(query, retriever, llm)
    print(f"A: {hyde_ans}")


if __name__ == "__main__":
    main()

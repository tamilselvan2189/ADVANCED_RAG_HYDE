# utils/query_rewriter.py
"""LLM-based query rewriting for improved retrieval recall."""
from __future__ import annotations

from utils.common import get_llm


SYSTEM_PROMPT = (
    "You are an expert at improving user queries for retrieval-based search. "
    "Rewrite the query to make it clearer, longer, and more specific, "
    "while keeping the original intent. Output ONLY the rewritten query."
)


def rewrite_query(user_query: str) -> str:
    """Return a rewritten/expanded version of the user query."""
    llm = get_llm()
    prompt = [
        ("system", SYSTEM_PROMPT),
        ("human", f"Rewrite the following query:\n\n{user_query}"),
    ]
    return llm.invoke(prompt).content.strip()

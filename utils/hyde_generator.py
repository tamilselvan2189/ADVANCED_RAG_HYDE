# utils/hyde_generator.py
"""
HyDE (Hypothetical Document Embeddings):
1. Ask the LLM to draft a plausible answer to the query.
2. Embed THAT synthetic answer (not the raw query).
3. Use the resulting vector for similarity search.
"""
from __future__ import annotations

import numpy as np

from utils.common import get_embedding_model, get_llm, load_config


HYDE_SYSTEM_PROMPT = (
    "Generate a short, factual-sounding hypothetical answer to help a retrieval "
    "system. Do NOT say you are guessing; produce a confident, concise paragraph."
)


def generate_hyde_answer(query: str) -> str:
    """Return a synthetic answer (string) for the given query."""
    llm = get_llm()
    prompt = [
        ("system", HYDE_SYSTEM_PROMPT),
        ("human", f"Question: {query}\n\nHypothetical Answer:"),
    ]
    return llm.invoke(prompt).content.strip()


def generate_hyde_embedding(query: str, verbose: bool = True) -> np.ndarray:
    """Run the full HyDE pipeline and return the embedding vector."""
    config = load_config()
    embedding_model = get_embedding_model(config)

    synthetic_answer = generate_hyde_answer(query)
    if verbose:
        print(f"\n🧪 Synthetic HyDE Answer:\n{synthetic_answer}\n")

    vector = embedding_model.embed_query(synthetic_answer)
    return np.array(vector, dtype=np.float32)

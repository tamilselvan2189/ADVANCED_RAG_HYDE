# utils/retriever.py
"""Build or load the FAISS retriever for the active provider."""
from __future__ import annotations

import os
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from utils.common import (
    get_embedding_model,
    get_index_path,
    load_config,
)


def create_retriever(chunks: List[Document], config: Optional[dict] = None):
    config = config or load_config()
    index_path = get_index_path(config)
    embedding_model = get_embedding_model(config)

    os.makedirs(index_path, exist_ok=True)
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(index_path)

    print(f"✅ Created new FAISS index at: {index_path}")
    return vectorstore.as_retriever(
        search_kwargs={"k": config["retrieval"]["top_k"]}
    )


def load_retriever(config: Optional[dict] = None):
    config = config or load_config()
    index_path = get_index_path(config)
    embedding_model = get_embedding_model(config)

    vectorstore = FAISS.load_local(
        index_path,
        embedding_model,
        allow_dangerous_deserialization=True,
    )
    print(f"✅ Loaded FAISS index from: {index_path}")
    return vectorstore.as_retriever(
        search_kwargs={"k": config["retrieval"]["top_k"]}
    )


def get_retriever(
    chunks_if_needed: Optional[List[Document]] = None,
    config: Optional[dict] = None,
):
    """Load the index if it exists, otherwise build it from the provided chunks."""
    config = config or load_config()
    index_path = get_index_path(config)

    # FAISS persists both `index.faiss` and `index.pkl`; check the actual file.
    if os.path.exists(os.path.join(index_path, "index.faiss")):
        return load_retriever(config)

    print(f"⚠️ No FAISS index at {index_path}. Building a new one...")
    if not chunks_if_needed:
        raise RuntimeError(
            "Index does not exist and no chunks were supplied to build one."
        )
    return create_retriever(chunks_if_needed, config)

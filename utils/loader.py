# utils/loader.py
"""Document loading and chunking."""
from __future__ import annotations

import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


SUPPORTED_EXTS = {".txt", ".pdf"}


def load_and_chunk_docs(
    folder_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[Document]:
    """Load every supported file from `folder_path` and split into chunks."""
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Docs folder not found: {folder_path}")

    docs: List[Document] = []
    for file in sorted(os.listdir(folder_path)):           # deterministic order
        path = os.path.join(folder_path, file)
        ext = os.path.splitext(file)[1].lower()

        if ext not in SUPPORTED_EXTS or os.path.getsize(path) == 0:
            continue

        if ext == ".txt":
            loader = TextLoader(path, encoding="utf-8")
        else:  # .pdf
            loader = PyPDFLoader(path)

        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)

    print(f"✅ Loaded {len(docs)} docs → {len(chunks)} chunks")
    return chunks

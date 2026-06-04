# utils/common.py
"""
Shared helpers: project paths, config loading, LLM & embedding factories.
Everything path-related is anchored to PROJECT_ROOT so the project is
independent of the current working directory.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")

# Load .env exactly once on import
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


def resolve_path(path: str) -> str:
    """Make a path absolute relative to the project root if it isn't already."""
    if not os.path.isabs(path):
        path = os.path.normpath(os.path.join(PROJECT_ROOT, path))
    return path


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_config(path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load and cache config.yaml. Cached so callers can invoke freely."""
    path = resolve_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"config.yaml not found at: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config.yaml is empty or invalid: {path}")
    return cfg


# --------------------------------------------------------------------------
# LLM / Embedding factories
# --------------------------------------------------------------------------
def get_llm(config: Dict[str, Any] | None = None):
    """Return a chat LLM instance based on config['llm']['provider']."""
    from langchain_openai import ChatOpenAI
    from langchain_google_genai import ChatGoogleGenerativeAI

    config = config or load_config()
    llm_cfg = config["llm"]
    provider = llm_cfg["provider"].lower()
    temperature = llm_cfg.get("temperature", 0.3)

    if provider == "openai":
        return ChatOpenAI(model=llm_cfg["model_openai"], temperature=temperature)
    if provider == "gemini":
        return ChatGoogleGenerativeAI(model=llm_cfg["model_gemini"], temperature=temperature)
    raise ValueError(f"Unsupported provider: {provider!r}. Use 'openai' or 'gemini'.")


def get_embedding_model(config: Dict[str, Any] | None = None):
    """Return an embedding model instance for the active provider."""
    from langchain_openai import OpenAIEmbeddings
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    config = config or load_config()
    provider = config["llm"]["provider"].lower()
    emb_cfg = config["embedding"]

    if provider == "openai":
        return OpenAIEmbeddings(model=emb_cfg["openai_model"])
    if provider == "gemini":
        return GoogleGenerativeAIEmbeddings(model=emb_cfg["gemini_model"])
    raise ValueError(f"Unsupported provider: {provider!r}.")


def get_index_path(config: Dict[str, Any] | None = None) -> str:
    """Return the absolute FAISS index path for the active provider."""
    config = config or load_config()
    provider = config["llm"]["provider"].lower()
    key = "faiss_openai" if provider == "openai" else "faiss_gemini"
    return resolve_path(config["vectordb"][key])

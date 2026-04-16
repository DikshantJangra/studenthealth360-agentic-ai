"""
Resource Factory — Singleton manager for shared heavy AI resources.
Ensures Groq LLM, Embeddings, and ChromaDB are loaded only once.
"""

from typing import Optional
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import (
    GROQ_API_KEY,
    GROQ_MODEL_NAME,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    EMBEDDING_MODEL_NAME,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
)

# ── Global singletons ────────────────────────────────────────────────
_llm: Optional[ChatGroq] = None
_embeddings: Optional[HuggingFaceEmbeddings] = None
_vectorstore: Optional[Chroma] = None


def get_llm() -> ChatGroq:
    """Get or initialize the Groq LLM instance."""
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL_NAME,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
    return _llm


def get_embeddings() -> HuggingFaceEmbeddings:
    """Get or initialize the HuggingFace embeddings model."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
        )
    return _embeddings


def get_vectorstore() -> Chroma:
    """Get or initialize the Chroma vector store connection."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=get_embeddings(),
            collection_name=CHROMA_COLLECTION_NAME,
        )
    return _vectorstore

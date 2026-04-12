"""
Build the ChromaDB vector store from medical guideline documents.
Run this once before starting the app:
    python rag/build_vectorstore.py
"""

import os
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

GUIDELINES_DIR = Path(__file__).resolve().parent / "guidelines"


def build_vectorstore():
    """Ingest, chunk, embed, and persist all guideline documents."""

    print(f"📂  Loading guidelines from: {GUIDELINES_DIR}")
    guideline_files = sorted(GUIDELINES_DIR.glob("*.txt"))

    if not guideline_files:
        print("⚠️  No .txt files found in guidelines directory!")
        return

    # ── Load documents ──────────────────────────────────────────────
    all_docs = []
    for fpath in guideline_files:
        loader = TextLoader(str(fpath), encoding="utf-8")
        docs = loader.load()
        # Tag each doc with its source filename for attribution
        for doc in docs:
            doc.metadata["source"] = fpath.name
        all_docs.extend(docs)
        print(f"   ✅  Loaded: {fpath.name}  ({len(docs)} document(s))")

    print(f"\n📄  Total documents loaded: {len(all_docs)}")

    # ── Chunk documents ─────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)
    print(f"✂️   Split into {len(chunks)} chunks  "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    # ── Embed & persist ─────────────────────────────────────────────
    print(f"\n🧠  Embedding with: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
    )

    # Remove existing store to rebuild cleanly
    if os.path.exists(CHROMA_PERSIST_DIR):
        import shutil
        shutil.rmtree(CHROMA_PERSIST_DIR)
        print(f"🗑️   Removed old vectorstore at: {CHROMA_PERSIST_DIR}")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION_NAME,
    )

    print(f"\n✅  Vectorstore built and persisted to: {CHROMA_PERSIST_DIR}")
    print(f"   Collection: {CHROMA_COLLECTION_NAME}")
    print(f"   Total chunks indexed: {len(chunks)}")

    return vectorstore


if __name__ == "__main__":
    build_vectorstore()

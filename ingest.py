"""
AgroAI Phase 3A — Knowledge Base Ingestion Pipeline
=====================================================
Builds a FAISS vector index from agriculture documents.

Supported formats (PDF-first architecture):
  • PDF    — PyPDFLoader  (page-level metadata preserved)
  • TXT    — TextLoader
  • MD     — TextLoader

Usage:
  python ingest.py               # Full rebuild
  python ingest.py --dry-run     # Preview chunks, no write
  python ingest.py --dir custom/ # Custom knowledge base dir

After running, restart Flask once so rag_service loads the new index.
"""

import os
import sys
import argparse
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ingest")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR  = os.path.join(BASE_DIR, "knowledge_base")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")

CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200

EMBED_MODEL = "all-MiniLM-L6-v2"   # Free, local, ~80 MB — no API key needed

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_documents(knowledge_dir: str) -> list:
    """
    Walk knowledge_dir recursively.
    Returns list of LangChain Document objects with metadata:
      { source, page, category, filename }
    """
    try:
        from langchain_community.document_loaders import PyPDFLoader, TextLoader
    except ImportError:
        logger.error("langchain-community not installed. Run: pip install langchain-community pypdf")
        sys.exit(1)

    all_docs = []
    total_files = 0
    skipped = 0

    for root, dirs, files in os.walk(knowledge_dir):
        # Derive category from immediate subdirectory name under knowledge_base
        rel = os.path.relpath(root, knowledge_dir)
        category = rel.split(os.sep)[0] if rel != "." else "general"

        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            fpath = os.path.join(root, fname)
            total_files += 1

            try:
                if ext == ".pdf":
                    loader = PyPDFLoader(fpath)
                    docs = loader.load()
                    # Enrich metadata per page
                    for doc in docs:
                        doc.metadata.update({
                            "source":   fname,
                            "category": category,
                            "filename": fname,
                            # page is already set by PyPDFLoader as doc.metadata["page"]
                        })
                else:
                    # TXT / MD
                    loader = TextLoader(fpath, encoding="utf-8")
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata.update({
                            "source":   fname,
                            "page":     0,
                            "category": category,
                            "filename": fname,
                        })

                all_docs.extend(docs)
                logger.info(f"  Loaded [{ext.upper()}] {category}/{fname} — {len(docs)} page(s)")

            except Exception as e:
                logger.warning(f"  Skipped {fname}: {e}")
                skipped += 1

    logger.info(f"\nTotal files loaded: {total_files - skipped} | Skipped: {skipped}")
    return all_docs


def split_documents(docs: list) -> list:
    """
    Split documents into overlapping chunks using RecursiveCharacterTextSplitter.
    Returns list of chunk Documents with metadata preserved.
    """
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", "---", ". ", " ", ""],
    )

    chunks = splitter.split_documents(docs)
    logger.info(f"Split {len(docs)} document pages → {len(chunks)} chunks "
                f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def build_vectorstore(chunks: list, vectorstore_dir: str) -> int:
    """
    Generate embeddings and save FAISS index to disk.
    Returns number of chunks indexed.
    """
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        logger.error("Missing deps. Run: pip install langchain-community faiss-cpu sentence-transformers")
        sys.exit(1)

    logger.info(f"Loading embedding model: {EMBED_MODEL} …")
    logger.info("(First run downloads ~80 MB — this may take a minute)")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # cosine similarity
    )

    logger.info(f"Generating embeddings for {len(chunks)} chunks …")
    t0 = time.time()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    elapsed = time.time() - t0
    logger.info(f"Embeddings generated in {elapsed:.1f}s")

    os.makedirs(vectorstore_dir, exist_ok=True)
    vectorstore.save_local(vectorstore_dir)
    logger.info(f"FAISS index saved → {vectorstore_dir}/")

    return len(chunks)


def run_ingestion(
    knowledge_dir: str = KNOWLEDGE_DIR,
    vectorstore_dir: str = VECTORSTORE_DIR,
    dry_run: bool = False,
) -> int:
    """
    Main ingestion entry point.
    Can be called from CLI or imported by app.py for the admin rebuild endpoint.
    Returns total chunks indexed (0 for dry run).
    """
    logger.info("=" * 55)
    logger.info("AgroAI RAG — Knowledge Base Ingestion Pipeline")
    logger.info("=" * 55)

    if not os.path.isdir(knowledge_dir):
        logger.error(f"knowledge_base/ directory not found: {knowledge_dir}")
        logger.error("Create the directory and add documents before running ingest.py")
        sys.exit(1)

    # Step 1: Load documents
    logger.info(f"\nStep 1/3 — Loading documents from: {knowledge_dir}")
    docs = load_documents(knowledge_dir)

    if not docs:
        logger.error("No documents found! Add PDFs or TXT files to knowledge_base/ subdirectories.")
        sys.exit(1)

    # Step 2: Split into chunks
    logger.info(f"\nStep 2/3 — Splitting into chunks …")
    chunks = split_documents(docs)

    if dry_run:
        logger.info("\n[DRY RUN] Chunks preview (first 3):")
        for i, chunk in enumerate(chunks[:3]):
            logger.info(f"  Chunk {i+1}: source={chunk.metadata.get('source')} | "
                       f"category={chunk.metadata.get('category')} | "
                       f"page={chunk.metadata.get('page')} | "
                       f"chars={len(chunk.page_content)}")
        logger.info(f"\n[DRY RUN] Total chunks that would be indexed: {len(chunks)}")
        logger.info("[DRY RUN] No FAISS index written.")
        return 0

    # Step 3: Build and save vectorstore
    logger.info(f"\nStep 3/3 — Building FAISS vectorstore …")
    chunk_count = build_vectorstore(chunks, vectorstore_dir)

    logger.info("\n" + "=" * 55)
    logger.info(f"✅ Ingestion complete!")
    logger.info(f"   Documents processed : {len(docs)}")
    logger.info(f"   Chunks indexed      : {chunk_count}")
    logger.info(f"   Vector store path   : {vectorstore_dir}/")
    logger.info(f"\nRestart Flask to reload the new FAISS index:")
    logger.info(f"   Ctrl+C → python app.py")
    logger.info("=" * 55)

    return chunk_count


# ── CLI Entry Point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AgroAI RAG — Build FAISS vector index from agriculture documents"
    )
    parser.add_argument(
        "--dir",
        default=KNOWLEDGE_DIR,
        help=f"Path to knowledge base directory (default: {KNOWLEDGE_DIR})",
    )
    parser.add_argument(
        "--vectorstore",
        default=VECTORSTORE_DIR,
        help=f"Path to save FAISS index (default: {VECTORSTORE_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview chunks without writing the FAISS index",
    )

    args = parser.parse_args()
    run_ingestion(
        knowledge_dir=args.dir,
        vectorstore_dir=args.vectorstore,
        dry_run=args.dry_run,
    )

"""
AgroAI Phase 3A — RAG Service
==============================
Singleton that loads the FAISS vector store once at Flask startup
and provides grounded Gemini answers with source citations.

Architecture:
  User Question
    → FAISS similarity search (Top-K chunks)
    → Relevance filter
    → Grounded Gemini prompt
    → Answer + Source metadata

Fallback:
  If no relevant chunks found (or index not built yet):
    → Direct Gemini answer, clearly marked as "General AI Response"
"""

import os
import logging
import time

logger = logging.getLogger(__name__)

BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")
EMBED_MODEL     = "all-MiniLM-L6-v2"

# FAISS L2 distance threshold — lower = more similar.
# For normalised embeddings: L2=0 (identical) … L2=1.41 (orthogonal).
# We require L2 < 1.2  ≈  cosine similarity > 0.28 to count as relevant.
RELEVANCE_THRESHOLD = 1.2
TOP_K = 5

# Working Gemini model (confirmed via API probe — gemini-1.5-flash was deprecated)
GEMINI_MODEL = "gemini-2.5-flash"


class RAGService:
    """
    Singleton RAG service. Call get_rag_service() to obtain the instance.
    """

    _instance = None

    def __init__(self):
        self._embeddings   = None
        self._vectorstore  = None
        self._ready        = False
        self._load()

    # ── Initialisation ──────────────────────────────────────────────────────

    def _load(self):
        """Load embedding model + FAISS index from disk."""
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self._embeddings = HuggingFaceEmbeddings(
                model_name=EMBED_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info(f"[RAG] Embedding model loaded: {EMBED_MODEL}")
        except Exception as e:
            logger.error(f"[RAG] Failed to load embedding model: {e}")
            return

        self._load_vectorstore()

    def _load_vectorstore(self):
        """Load FAISS index from vectorstore/ directory."""
        if not os.path.isdir(VECTORSTORE_DIR):
            logger.warning(
                "[RAG] vectorstore/ not found. "
                "Run: python ingest.py  to build the index."
            )
            self._ready = False
            return

        try:
            from langchain_community.vectorstores import FAISS
            self._vectorstore = FAISS.load_local(
                VECTORSTORE_DIR,
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
            self._ready = True
            chunk_count = self._vectorstore.index.ntotal if hasattr(self._vectorstore, 'index') else '?'
            logger.info(
                f"[RAG] FAISS vectorstore loaded from: {VECTORSTORE_DIR} | "
                f"Chunks indexed: {chunk_count}"
            )
            # ── CHECK 1: RAG Service Health ──────────────────────────────────
            logger.info(
                f"\n[RAG DEBUG] Service health check:\n"
                f"  Ready       : {self._ready}\n"
                f"  Vectorstore : {type(self._vectorstore).__name__}\n"
                f"  Chunk Count : {chunk_count}\n"
                f"  Model       : {GEMINI_MODEL}"
            )
        except Exception as e:
            logger.error(f"[RAG] Failed to load vectorstore: {e}")
            self._ready = False

    def reload(self):
        """Hot-reload vectorstore after admin rebuild. Reloads embeddings too."""
        logger.info("[RAG] Reloading vectorstore …")
        self._load_vectorstore()

    def is_ready(self) -> bool:
        return self._ready

    # ── Core RAG Method ─────────────────────────────────────────────────────

    def answer_agriculture_query(self, question: str, language: str = "en") -> dict:
        """
        Main entry point for the RAG pipeline.

        Args:
            question: User's agriculture question
            language: "en" (English) or "hi" (Hindi)

        Returns:
            {
              "answer":  str,
              "sources": [{"document": str, "page": int, "category": str}],
              "mode":    "rag" | "fallback"
            }
        """
        question = (question or "").strip()
        if not question:
            return {"answer": "Please ask a question.", "sources": [], "mode": "error"}

        # ── Step 1: Try vector search ─────────────────────────────────────
        chunks = []
        if self._ready and self._vectorstore:
            try:
                raw = self._vectorstore.similarity_search_with_score(question, k=TOP_K)
                # Filter by relevance threshold (L2 distance)
                chunks = [(doc, score) for doc, score in raw if score < RELEVANCE_THRESHOLD]
                # ── CHECK 2: Vector Search debug ─────────────────────────────────
                all_scores = [round(float(s), 4) for _, s in raw]
                logger.info(
                    f"\n[RAG DEBUG] Vector search for: {question[:60]!r}\n"
                    f"  Raw Hits         : {len(raw)}\n"
                    f"  Relevant Chunks  : {len(chunks)}\n"
                    f"  All Scores (L2)  : {all_scores}\n"
                    f"  Threshold (L2<)  : {RELEVANCE_THRESHOLD}"
                )
            except Exception as e:
                logger.error(f"[RAG] Vector search failed: {e}")

        # ── Step 2: Choose RAG or Fallback ────────────────────────────────
        # ── CHECK 5: Mode selection ───────────────────────────────────────
        selected_mode = "rag" if chunks else "fallback"
        logger.info(f"[RAG DEBUG] Selected Mode: {selected_mode}")

        if chunks:
            return self._rag_answer(question, chunks, language)
        else:
            logger.info("[RAG] No relevant chunks — falling back to direct Gemini")
            return self._fallback_answer(question, language)

    # ── RAG Answer (Grounded) ───────────────────────────────────────────────

    def _rag_answer(self, question: str, chunks: list, language: str) -> dict:
        """Build grounded Gemini prompt from retrieved chunks."""
        # Build context string
        context_parts = []
        for i, (doc, score) in enumerate(chunks, 1):
            src  = doc.metadata.get("source", "Unknown")
            pg   = doc.metadata.get("page", 0)
            cat  = doc.metadata.get("category", "general")
            context_parts.append(
                f"[Source {i}: {src} | Category: {cat} | Page: {pg}]\n"
                f"{doc.page_content.strip()}"
            )
        context = "\n\n---\n\n".join(context_parts)

        # Language instruction
        lang_instruction = (
            "CRITICAL: Respond ENTIRELY in Hindi (हिंदी). "
            "Use simple, farmer-friendly Hindi. "
            "Technical terms may remain in English.\n\n"
            if language == "hi" else ""
        )

        system_prompt = (
            f"{lang_instruction}"
            "You are an Agriculture Expert Assistant for Indian farmers, powered by verified ICAR documents.\n\n"
            "RULES:\n"
            "1. Answer ONLY using the provided context below.\n"
            "2. Be specific — include dosages, timings, and product names from the context.\n"
            "3. Use farmer-friendly language with bullet points.\n"
            "4. If the answer is not in the context, say exactly: "
            "\"I could not find this information in the agriculture knowledge base.\"\n"
            "5. NEVER add information not present in the context.\n\n"
            f"CONTEXT FROM KNOWLEDGE BASE:\n{context}"
        )

        answer = self._call_gemini(system_prompt, question)

        # Build deduplicated source list
        seen = set()
        sources = []
        for doc, _ in chunks:
            key = (doc.metadata.get("source", ""), doc.metadata.get("page", 0))
            if key not in seen:
                seen.add(key)
                sources.append({
                    "document": doc.metadata.get("source", "Unknown Document"),
                    "page":     int(doc.metadata.get("page", 0)) + 1,  # 1-indexed for display
                    "category": doc.metadata.get("category", "general"),
                })

        return {"answer": answer, "sources": sources, "mode": "rag"}

    # ── Fallback Answer (Direct Gemini, no grounding) ───────────────────────

    def _fallback_answer(self, question: str, language: str) -> dict:
        """
        Direct Gemini call when no relevant knowledge base chunks found.
        Clearly marks response as general AI (not from knowledge base).
        """
        lang_instruction = (
            "CRITICAL: Respond ENTIRELY in Hindi (हिंदी). "
            "Use simple, farmer-friendly Hindi.\n\n"
            if language == "hi" else ""
        )

        system_prompt = (
            f"{lang_instruction}"
            "You are a helpful agriculture assistant. "
            "Answer the farmer's question to the best of your knowledge. "
            "Be concise, practical, and use bullet points. "
            "Focus on Indian farming conditions and practices."
        )

        answer = self._call_gemini(system_prompt, question)

        # Prepend disclaimer
        disclaimer = (
            "⚠️ *सामान्य AI उत्तर (कृषि ज्ञान आधार से नहीं)*\n\n"
            if language == "hi" else
            "⚠️ *General AI Response — not from the Agriculture Knowledge Base*\n\n"
        )

        return {
            "answer":  disclaimer + answer,
            "sources": [],
            "mode":    "fallback",
        }

    # ── Gemini Call ─────────────────────────────────────────────────────────

    def _call_gemini(self, system_prompt: str, user_message: str, retries: int = 1) -> str:
        """
        Call Gemini (model: gemini-2.5-flash) with retry on quota/server errors.
        gemini-1.5-flash was deprecated and removed from the API in 2025.
        """
        try:
            import google.generativeai as genai
        except ImportError:
            return "Gemini SDK not installed. Run: pip install google-generativeai"

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return "GOOGLE_API_KEY not configured. Please set it in your .env file."

        # ── CHECK 3: Gemini call info ─────────────────────────────────────
        logger.info(
            f"\n[RAG DEBUG] Gemini call\n"
            f"  API key detected : True ({api_key[:8]}...)\n"
            f"  Model name       : {GEMINI_MODEL}"
        )

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,   # ← was 'gemini-1.5-flash' (DEAD)
            system_instruction=system_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=1024,
                temperature=0.3,   # Lower temp for factual RAG answers
            ),
        )

        last_err = None
        for attempt in range(retries + 1):
            try:
                response = model.generate_content(user_message)
                text = getattr(response, "text", None)
                if not text:
                    raise ValueError("Empty response from Gemini")
                logger.info(f"[RAG] Gemini responded successfully ({len(text)} chars)")
                return text.strip()
            except Exception as e:
                last_err = e
                err_str = str(e).upper()
                is_retriable = any(
                    kw in err_str
                    for kw in ["429", "QUOTA", "RESOURCE_EXHAUSTED", "503", "500", "UNAVAILABLE"]
                )
                if attempt < retries and is_retriable:
                    logger.warning(f"[RAG] Gemini attempt {attempt+1} failed, retrying in 5s: {e}")
                    time.sleep(5)
                else:
                    break

        # ── Surface the EXACT exception so it's visible in logs ──────────
        logger.error(
            f"\n[RAG DEBUG] Gemini call FAILED after {retries+1} attempt(s)\n"
            f"  Exception type : {type(last_err).__name__}\n"
            f"  Exception text : {last_err}"
        )
        # Return the actual error message so it's visible in the UI during debugging
        return (
            f"I'm unable to reach the AI service right now. "
            f"Please try again in a moment. "
            f"[DEBUG: {type(last_err).__name__}: {str(last_err)[:120]}]"
        )


# ── Singleton Accessor ──────────────────────────────────────────────────────

_rag_instance: RAGService | None = None


def get_rag_service() -> RAGService:
    """
    Return the singleton RAGService instance.
    Initialises once on first call (loads embeddings + FAISS from disk).
    Subsequent calls return the cached instance — no repeated I/O.
    """
    global _rag_instance
    if _rag_instance is None:
        logger.info("[RAG] Initialising RAG service …")
        _rag_instance = RAGService()
    return _rag_instance

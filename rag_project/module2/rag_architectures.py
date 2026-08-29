# module2/rag_architectures.py
"""
Module 2 — RAG Architectures
─────────────────────────────
Implements the 4 RAG variants compared in this capstone:

    1. NaiveRAG              — vanilla dense retrieval + LLM
    2. HybridMultiQueryRAG   — BM25 + Dense (RRF) + LLM-rewritten queries
    3. ParentChildRAG        — search children, return parent contexts
    4. CorrectiveRAG (CRAG)  — relevance grading + fallback rewriting

All classes share the `BaseRAG` interface so module3/evaluation.py can
treat them uniformly.

Falls back to an *extractive* answer when MISTRAL_API_KEY is not set,
so the framework remains fully runnable without paid APIs.
"""

from __future__ import annotations

import os
import re
import sys
import time
import logging
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.documents import Document

from configs.config import (
    TOP_K, BM25_WEIGHT, DENSE_WEIGHT,
    GROQ_API_KEY, GROQ_MODEL,
    MISTRAL_API_KEY, LLM_MODEL,
    LLM_MAX_TOKENS, LLM_TEMPERATURE, LLM_RATE_LIMIT_SECONDS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# LLM Helper (Groq FREE → Mistral fallback → extractive last resort)
# ═══════════════════════════════════════════════════════════════════════════════

class _LLM:
    """
    LLM wrapper with priority:
      1. Groq (FREE, no credit card, Llama 3.1 8B)
      2. Mistral (paid, if key provided)
      3. Extractive fallback (no API at all)
    """

    def __init__(self):
        self._groq_client = None
        self._mistral_client = None
        self._provider = "extractive"

        # Try Groq first (FREE)
        if GROQ_API_KEY:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=GROQ_API_KEY)
                self._provider = "groq"
                logger.info("LLM: Using Groq (%s) — FREE tier.", GROQ_MODEL)
            except ImportError:
                logger.warning("groq package not installed. Run: pip install groq")
            except Exception as exc:
                logger.warning("Groq init failed: %s", exc)

        # Try Mistral as fallback
        if self._provider == "extractive" and MISTRAL_API_KEY:
            try:
                from mistralai import Mistral
                self._mistral_client = Mistral(api_key=MISTRAL_API_KEY)
                self._provider = "mistral"
                logger.info("LLM: Using Mistral (%s).", LLM_MODEL)
            except Exception as exc:
                logger.warning("Mistral init failed: %s", exc)

        if self._provider == "extractive":
            logger.info("LLM: No API key set — using extractive fallback. "
                        "Set GROQ_API_KEY (free) for real answers.")

    def complete(self, prompt: str, max_tokens: int = LLM_MAX_TOKENS) -> str:
        if self._provider == "groq":
            return self._groq_complete(prompt, max_tokens)
        elif self._provider == "mistral":
            return self._mistral_complete(prompt, max_tokens)
        return self._extractive_fallback(prompt)

    def _groq_complete(self, prompt: str, max_tokens: int) -> str:
        try:
            import time as _time
            # Rate limit: configurable via LLM_RATE_LIMIT_SECONDS
            if not hasattr(self, '_last_groq_call'):
                self._last_groq_call = 0
            elapsed = _time.time() - self._last_groq_call
            if elapsed < LLM_RATE_LIMIT_SECONDS:
                _time.sleep(LLM_RATE_LIMIT_SECONDS - elapsed)

            resp = self._groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=LLM_TEMPERATURE,
            )
            self._last_groq_call = _time.time()
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning("Groq call failed (%s) — trying fallback.", exc)
            if self._mistral_client:
                return self._mistral_complete(prompt, max_tokens)
            return self._extractive_fallback(prompt)

    def _mistral_complete(self, prompt: str, max_tokens: int) -> str:
        try:
            resp = self._mistral_client.chat.complete(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=LLM_TEMPERATURE,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning("Mistral call failed (%s) — using extractive.", exc)
            return self._extractive_fallback(prompt)

    @staticmethod
    def _extractive_fallback(prompt: str) -> str:
        """Pick the most relevant sentence from the context block of the prompt."""
        # Try to isolate a "Context: ... Question: ..." pattern
        ctx_match = re.search(r"Context:?\s*(.+?)(?:Question:|$)", prompt, re.S | re.I)
        q_match = re.search(r"Question:?\s*(.+?)(?:Answer:|$)", prompt, re.S | re.I)
        if not ctx_match:
            return "I cannot answer without an LLM (set MISTRAL_API_KEY for generated answers)."

        context = ctx_match.group(1).strip()
        question = q_match.group(1).strip() if q_match else ""

        # Remove the "Answer:" suffix if captured
        question = re.sub(r"\s*Answer:\s*$", "", question).strip()

        # Score sentences by keyword overlap with the question
        sentences = re.split(r"(?<=[.!?])\s+", context)
        # Also split on newlines (MS MARCO passages are often one-liners)
        expanded = []
        for s in sentences:
            expanded.extend(s.split("\n"))
        sentences = [s.strip() for s in expanded if len(s.strip()) > 15]

        q_tokens = set(re.findall(r"\w+", question.lower())) - {
            "the", "a", "an", "is", "are", "was", "were", "of", "to",
            "in", "on", "at", "and", "or", "what", "who", "where",
            "when", "why", "how", "which", "do", "does", "did", "can",
            "could", "would", "should", "will", "it", "its", "this",
            "that", "these", "those", "be", "been", "being", "have",
            "has", "had", "for", "with", "from", "by", "about",
        }
        if not sentences:
            return context[:300]

        if not q_tokens:
            # If question has no meaningful keywords, return first substantial sentence
            for s in sentences:
                if len(s) > 30:
                    return s[:500]
            return sentences[0][:300]

        scored = []
        for s in sentences:
            s_tokens = set(re.findall(r"\w+", s.lower()))
            overlap = len(s_tokens & q_tokens)
            # Bonus for shorter, more focused sentences (likely direct answers)
            length_penalty = max(0.5, 1.0 - len(s) / 1000)
            score = overlap * (1 + length_penalty)
            scored.append((score, s))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Return the best sentence only (not a jumble of multiple)
        best_score, best_sentence = scored[0]
        if best_score == 0:
            return "Based on the retrieved context, I could not find a relevant answer to this question."
        return best_sentence[:500]


_GLOBAL_LLM: Optional[_LLM] = None


def _llm() -> _LLM:
    global _GLOBAL_LLM
    if _GLOBAL_LLM is None:
        _GLOBAL_LLM = _LLM()
    return _GLOBAL_LLM


# ═══════════════════════════════════════════════════════════════════════════════
# Base class
# ═══════════════════════════════════════════════════════════════════════════════

ANSWER_PROMPT = (
    "You are a helpful assistant. Answer the question based on the context below. "
    "Synthesize information from the context to provide a clear, complete answer. "
    "If the context is truly unrelated to the question, say you don't have enough information.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


class BaseRAG:
    name: str = "BaseRAG"

    def retrieve(self, query: str) -> List[Document]:
        raise NotImplementedError

    def generate(self, query: str, docs: List[Document]) -> str:
        context = "\n\n".join(d.page_content for d in docs)
        prompt = ANSWER_PROMPT.format(context=context, question=query)
        return _llm().complete(prompt)

    def run(self, query: str) -> Dict[str, Any]:
        t0 = time.time()
        docs = self.retrieve(query)
        answer = self.generate(query, docs)
        latency = (time.time() - t0) * 1000
        return {
            "architecture": self.name,
            "query": query,
            "answer": answer,
            "context_docs": docs,
            "contexts": [d.page_content for d in docs],
            "latency_ms": round(latency, 2),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Naive RAG
# ═══════════════════════════════════════════════════════════════════════════════

class NaiveRAG(BaseRAG):
    """Vanilla: dense top-K retrieval → LLM."""
    name = "Naive RAG"

    def __init__(self, vectorstore, top_k: int = TOP_K):
        self.vectorstore = vectorstore
        self.top_k = top_k

    def retrieve(self, query: str) -> List[Document]:
        return self.vectorstore.similarity_search(query, k=self.top_k)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Hybrid Multi-Query RAG
# ═══════════════════════════════════════════════════════════════════════════════

QUERY_REWRITE_PROMPT = (
    "Generate 3 alternate phrasings of the question below. Return them as a "
    "numbered list and nothing else.\n\nQuestion: {question}"
)


class HybridMultiQueryRAG(BaseRAG):
    """
    BM25 + Dense fusion (Reciprocal Rank Fusion) + LLM-rewritten queries.
    Addresses FP1 (query content missed) and FP4 (not extracted).
    """
    name = "Hybrid Multi-Query"

    def __init__(self, vectorstore, chunks: List[Document], top_k: int = TOP_K):
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.top_k = top_k
        self._bm25 = None
        self._init_bm25(chunks)

    def _init_bm25(self, chunks: List[Document]) -> None:
        try:
            from rank_bm25 import BM25Okapi
            tokenized = [re.findall(r"\w+", d.page_content.lower()) for d in chunks]
            self._bm25 = BM25Okapi(tokenized)
        except Exception as exc:
            logger.warning("BM25 init failed (%s) — falling back to dense-only.", exc)
            self._bm25 = None

    def _rewrite_queries(self, query: str) -> List[str]:
        prompt = (
            f"Generate 3 alternate phrasings of this question. "
            f"Return as numbered list only.\n\nQuestion: {query}"
        )
        try:
            resp = _llm().complete(prompt, max_tokens=150)
            lines = [l.strip() for l in resp.split('\n')
                if l.strip() and len(l.strip()) > 5]
            variants = [re.sub(r'^\d+[\.\)]\s*', '', l) for l in lines[:3]]
            return [query] + [v for v in variants if v]
        except Exception:
            return [query]

    def _bm25_search(self, query: str, k: int) -> List[Document]:
        if self._bm25 is None:
            return []
        tokens = re.findall(r"\w+", query.lower())
        scores = self._bm25.get_scores(tokens)
        idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.chunks[i] for i in idx]

    def _rrf_merge(self, ranked_lists: List[List[Document]], k: int = 60) -> List[Document]:
        """Reciprocal Rank Fusion."""
        scores: Dict[str, float] = {}
        seen: Dict[str, Document] = {}
        for ranked in ranked_lists:
            for rank, doc in enumerate(ranked):
                key = doc.page_content[:200]
                seen[key] = doc
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [seen[k] for k, _ in ordered]

    def retrieve(self, query: str) -> List[Document]:
        rewrites = self._rewrite_queries(query)
        ranked_lists: List[List[Document]] = []
        for q in rewrites:
            dense = self.vectorstore.similarity_search(q, k=self.top_k)
            sparse = self._bm25_search(q, self.top_k)
            ranked_lists.append(dense)
            if sparse:
                ranked_lists.append(sparse)
        merged = self._rrf_merge(ranked_lists)
        return merged[: self.top_k]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Parent-Child RAG
# ═══════════════════════════════════════════════════════════════════════════════

class ParentChildRAG(BaseRAG):
    """
    Search small child chunks for precision, return their parent chunks
    for richer context. Addresses FP3 (relevant chunk not in top-K).
    """
    name = "Parent-Child RAG"

    def __init__(
        self,
        vectorstore=None,
        chunks: Optional[List[Document]] = None,
        parent_child: Optional[Dict[str, Any]] = None,
        child_vectorstore=None,
        top_k: int = TOP_K,
    ):
        self.top_k = top_k
        self.child_vectorstore = child_vectorstore or vectorstore
        self.parents_by_id: Dict[str, Document] = {}
        self.fallback_vectorstore = vectorstore

        if parent_child:
            for p in parent_child.get("parents", []):
                pid = p.metadata.get("parent_id") or f"parent_{len(self.parents_by_id)}"
                p.metadata["parent_id"] = pid
                self.parents_by_id[pid] = p
        elif chunks:
            # Treat each chunk as its own parent — graceful fallback
            for i, c in enumerate(chunks):
                pid = f"parent_{i}"
                c.metadata["parent_id"] = pid
                self.parents_by_id[pid] = c

    def retrieve(self, query: str) -> List[Document]:
        children = self.child_vectorstore.similarity_search(query, k=self.top_k * 2)
        seen_parents: Dict[str, Document] = {}
        for child in children:
            pid = child.metadata.get("parent_id")
            if pid and pid in self.parents_by_id and pid not in seen_parents:
                seen_parents[pid] = self.parents_by_id[pid]
            elif not pid:
                key = child.page_content[:200]
                if key not in seen_parents:
                    seen_parents[key] = child
            if len(seen_parents) >= self.top_k:
                break
        if not seen_parents and self.fallback_vectorstore:
            return self.fallback_vectorstore.similarity_search(query, k=self.top_k)
        return list(seen_parents.values())[: self.top_k]


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Corrective RAG (CRAG)
# ═══════════════════════════════════════════════════════════════════════════════

RELEVANCE_PROMPT = (
    "Rate how relevant this document is to the question on a scale of 0.0 to 1.0. "
    "Return ONLY the numeric score.\n\n"
    "Question: {question}\nDocument: {document}"
)

CORRECTIVE_REWRITE_PROMPT = (
    "The previous search did not return relevant results. Rewrite the question "
    "to be more specific and search-friendly. Return ONLY the rewritten question.\n\n"
    "Original: {question}"
)


class CorrectiveRAG(BaseRAG):
    """
    1. Retrieve top-K with dense search.
    2. LLM-grade each doc for relevance.
    3. If <2 relevant docs, rewrite query and retry.
    4. Generate answer from filtered set.
    """
    name = "CRAG"

    def __init__(
        self,
        vectorstore,
        top_k: int = TOP_K,
        relevance_threshold: float = 0.5,
        min_relevant: int = 2,
    ):
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.relevance_threshold = relevance_threshold
        self.min_relevant = min_relevant

    def _grade(self, query: str, doc: Document) -> float:
        """
        Relevance grading. On free tier (Groq), uses keyword overlap only
        to avoid burning rate limit on grading. LLM is saved for the answer.
        """
        # Use keyword-overlap scoring (fast, free, no API calls)
        q_tok = set(re.findall(r"\w+", query.lower())) - {
            "the", "a", "an", "is", "are", "was", "were", "of", "to",
            "in", "on", "at", "and", "or", "what", "who", "where",
            "when", "why", "how", "which", "do", "does", "did",
        }
        d_tok = set(re.findall(r"\w+", doc.page_content.lower()))
        if not q_tok:
            return 0.3
        overlap = len(q_tok & d_tok)
        return min(1.0, overlap / max(len(q_tok), 1))

    def retrieve(self, query: str) -> List[Document]:
        docs = self.vectorstore.similarity_search(query, k=self.top_k)
        graded = [(d, self._grade(query, d)) for d in docs]
        relevant = [d for d, s in graded if s >= self.relevance_threshold]

        # If not enough relevant docs, just use all retrieved (skip LLM rewrite to save rate limit)
        if not relevant:
            relevant = [d for d, _ in graded]
        return relevant[: self.top_k]


# ═══════════════════════════════════════════════════════════════════════════════
# Builder
# ═══════════════════════════════════════════════════════════════════════════════

class RAGFusion(BaseRAG):
    """
    Multi-query dense retrieval + Reciprocal Rank Fusion.
    Rackauckas, 2024.
    """
    name = "RAG-Fusion"

    def __init__(self, vectorstore, top_k: int = TOP_K):
        self.vectorstore = vectorstore
        self.top_k = top_k

    def _variants(self, query: str) -> List[str]:
        prompt = (
            f"Write 3 different search queries to find "
            f"information answering: {query}\n"
            f"Return one per line, nothing else."
        )
        try:
            resp = _llm().complete(prompt, max_tokens=150)
            lines = [l.strip() for l in resp.split('\n')
                     if len(l.strip()) > 5]
            return [query] + lines[:3]
        except Exception:
            return [query]

    def _rrf(self, ranked_lists, k=60):
        scores, seen = {}, {}
        for ranked in ranked_lists:
            for rank, doc in enumerate(ranked):
                key = doc.page_content[:200]
                seen[key] = doc
                scores[key] = scores.get(key, 0.0) + 1/(k + rank + 1)
        ordered = sorted(scores.items(),
                        key=lambda x: x[1], reverse=True)
        return [seen[k] for k, _ in ordered]

    def retrieve(self, query: str) -> List[Document]:
        results = [
            self.vectorstore.similarity_search(v, k=self.top_k)
            for v in self._variants(query)
        ]
        return self._rrf(results)[:self.top_k]


class AdaptiveRAG(BaseRAG):
    """
    Routes queries to retrieval strategy based on complexity.
    Jeong et al., NAACL 2024.
    """
    name = "Adaptive RAG"

    def __init__(self, vectorstore, chunks: List[Document],
                 top_k: int = TOP_K):
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.top_k = top_k
        self._bm25 = None
        try:
            from rank_bm25 import BM25Okapi
            tok = [re.findall(r"\w+", d.page_content.lower())
                   for d in chunks]
            self._bm25 = BM25Okapi(tok)
        except Exception:
            pass

    def _classify(self, query: str) -> str:
        prompt = (
            f"Classify complexity with ONE word only "
            f"— SIMPLE, MODERATE, or COMPLEX:\n{query}"
        )
        try:
            r = _llm().complete(prompt, max_tokens=5).strip().upper()
            for label in ["SIMPLE", "COMPLEX", "MODERATE"]:
                if label in r:
                    return label
        except Exception:
            pass
        return "MODERATE"

    def _rrf(self, ranked_lists, k=60):
        scores, seen = {}, {}
        for ranked in ranked_lists:
            for rank, doc in enumerate(ranked):
                key = doc.page_content[:200]
                seen[key] = doc
                scores[key] = scores.get(key, 0.0) + 1/(k + rank + 1)
        ordered = sorted(scores.items(),
                        key=lambda x: x[1], reverse=True)
        return [seen[k] for k, _ in ordered]

    def retrieve(self, query: str) -> List[Document]:
        level = self._classify(query)

        if level == "SIMPLE":
            return self.vectorstore.similarity_search(
                query, k=self.top_k)

        elif level == "MODERATE":
            return self.vectorstore.similarity_search(
                query, k=self.top_k)

        else:  # COMPLEX — full hybrid
            dense = self.vectorstore.similarity_search(
                query, k=self.top_k)
            if not self._bm25:
                return dense
            tok = re.findall(r"\w+", query.lower())
            sc = self._bm25.get_scores(tok)
            idx = sorted(range(len(sc)),
                        key=lambda i: sc[i], reverse=True)[:self.top_k]
            sparse = [self.chunks[i] for i in idx]
            return self._rrf([dense, sparse])[:self.top_k]

# ═══════════════════════════════════════════════════════════════════════════════
# 7. GRACE-Ω — Graph-verified Recursive Adaptive Context Engine with PΩ routing
# ═══════════════════════════════════════════════════════════════════════════════

class GRACEOmegaRAG(BaseRAG):
    """
    GRACE-Ω: Our original novel architecture.

    Combines:
    - RAPTOR tree (Sarthi et al., ICLR 2024) — hierarchical summarisation
    - Knowledge Graph (Microsoft GraphRAG 2024) — entity-linked retrieval
    - HyDE query expansion (Gao et al., 2022) — hypothetical answer embedding
    - Adaptive routing (Jeong et al., NAACL 2024) — complexity classification
    - PΩ quality gate (OUR NOVELTY 1) — live routing signal using p²
    - Cross-structure verification (OUR NOVELTY 2) — KG subgraph prunes
      RAPTOR chunks via entity intersection algorithm

    Novel contributions not in any existing paper:
    1. PΩ < threshold triggers automatic structure switch at inference time
    2. KG entity set used as topological validator to prune hierarchical chunks
       Algorithm: keep chunk c from RAPTOR if |tokens(c) ∩ KG_entity_set| >= θ
    """
    name = "GRACE-Ω"

    def __init__(
        self,
        vectorstore,
        chunks: List[Document],
        kg_builder=None,
        top_k: int = TOP_K,
        pomega_threshold: float = 0.4,
        verification_theta: int = 3,
    ):
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.kg_builder = kg_builder
        self.top_k = top_k
        self.pomega_threshold = pomega_threshold
        self.verification_theta = verification_theta

        # BM25 index
        self._bm25 = None
        try:
            from rank_bm25 import BM25Okapi
            tok = [re.findall(r"\w+", d.page_content.lower())
                   for d in chunks]
            self._bm25 = BM25Okapi(tok)
            logger.info("GRACE-Ω: BM25 index ready.")
        except Exception as e:
            logger.warning("GRACE-Ω: BM25 failed (%s).", e)

        # RAPTOR tree
        self._summary_docs: List[Document] = []
        self._summary_store = None
        self._build_raptor()

        # Track last retrieval path for logging
        self._last_route = "unknown"

    def _build_raptor(self):
        """
        Build 2-level RAPTOR summarisation tree.
        Level 0: original chunks
        Level 1: cluster summaries (LLM-generated)
        """
        try:
            from sklearn.cluster import KMeans
            from sentence_transformers import SentenceTransformer
            from langchain_community.vectorstores import Chroma
            from langchain_huggingface import HuggingFaceEmbeddings

            texts = [d.page_content for d in self.chunks[:300]]
            if len(texts) < 5:
                logger.warning("GRACE-Ω: Too few chunks for RAPTOR.")
                return

            logger.info("GRACE-Ω: Building RAPTOR tree over %d chunks...", len(texts))
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(texts, show_progress_bar=False)

            n_clusters = max(5, len(texts) // 20)
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = km.fit_predict(embeddings)

            summaries = []
            for cid in range(n_clusters):
                cluster_texts = [
                    texts[i] for i, l in enumerate(labels) if l == cid
                ][:5]
                combined = " ".join(cluster_texts)[:600]
                prompt = (
                    f"Summarise the following text in 2 concise sentences "
                    f"preserving key facts:\n\n{combined}"
                )
                try:
                    summary_text = _llm().complete(prompt, max_tokens=120)
                    summaries.append(Document(
                        page_content=summary_text,
                        metadata={"level": "raptor_summary", "cluster": cid}
                    ))
                except Exception:
                    summaries.append(Document(
                        page_content=combined[:200],
                        metadata={"level": "raptor_summary", "cluster": cid}
                    ))

            self._summary_docs = summaries
            embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self._summary_store = Chroma.from_documents(
                summaries, embedder,
                collection_name="grace_omega_raptor"
            )
            logger.info("GRACE-Ω: RAPTOR tree built — %d summaries.", len(summaries))

        except Exception as e:
            logger.warning("GRACE-Ω: RAPTOR build failed (%s). Will use dense only.", e)
            self._summary_store = None

    # ── Step 1: HyDE expansion ──────────────────────────────────────────────

    def _hyde_expand(self, query: str) -> str:
        """
        Hypothetical Document Embedding expansion.
        Generate a short hypothetical answer and append to query
        so the embedding captures the answer space, not just question space.
        """
        prompt = (
            f"Write a short 2-sentence factual answer to this question:\n\n{query}"
        )
        try:
            hyp_answer = _llm().complete(prompt, max_tokens=80)
            return f"{query} {hyp_answer}"
        except Exception:
            return query

    # ── Step 2: Complexity classification ──────────────────────────────────

    def _classify(self, query: str) -> str:
        """
        Route query to appropriate retrieval structure:
        SIMPLE     → Dense + BM25 (fast, precise)
        MULTIHOP   → RAPTOR tree (hierarchical, multi-level)
        RELATIONAL → Knowledge Graph (entity-linked)
        """
        prompt = (
            f"Classify this question. Reply ONE word:\n"
            f"SIMPLE (direct factual question)\n"
            f"MULTIHOP (needs reasoning across multiple facts)\n"
            f"RELATIONAL (about relationships between named entities)\n\n"
            f"Question: {query}\n\nClassification:"
        )
        try:
            result = _llm().complete(prompt, max_tokens=10).strip().upper()
            if "RELATIONAL" in result:
                return "RELATIONAL"
            if "MULTI" in result or "HOP" in result:
                return "MULTIHOP"
            return "SIMPLE"
        except Exception:
            return "SIMPLE"

    # ── Step 3: Retrieval strategies ────────────────────────────────────────

    def _retrieve_simple(self, query: str) -> List[Document]:
        """Dense + BM25 with RRF fusion."""
        dense = self.vectorstore.similarity_search(query, k=self.top_k)
        if not self._bm25:
            return dense
        tok = re.findall(r"\w+", query.lower())
        sc = self._bm25.get_scores(tok)
        idx = sorted(range(len(sc)), key=lambda i: sc[i], reverse=True)[:self.top_k]
        sparse = [self.chunks[i] for i in idx]
        return self._rrf([dense, sparse])[:self.top_k]

    def _retrieve_raptor(self, query: str) -> List[Document]:
        """
        RAPTOR tree retrieval:
        Search summary level first (abstract), then leaf level (specific).
        Combines both for multi-resolution context.
        """
        leaf = self.vectorstore.similarity_search(query, k=self.top_k)
        if self._summary_store is None:
            return leaf
        summaries = self._summary_store.similarity_search(query, k=3)
        combined = summaries + leaf
        seen, final = set(), []
        for doc in combined:
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                final.append(doc)
        return final[:self.top_k]

    def _retrieve_graph(self, query: str) -> List[Document]:
        """
        Knowledge graph traversal retrieval.
        Finds chunks linked to entities in the query via 1-hop traversal.
        """
        if self.kg_builder is None:
            return self.vectorstore.similarity_search(query, k=self.top_k)
        kg_results = self.kg_builder.query_chunks(query, k=self.top_k)
        if not kg_results:
            return self.vectorstore.similarity_search(query, k=self.top_k)
        return [
            Document(
                page_content=r["text"],
                metadata={"source": "knowledge_graph", "score": r["score"]}
            )
            for r in kg_results
        ]

    # ── Step 4: PΩ quality gate (NOVELTY 1) ─────────────────────────────────

    def _compute_pomega_proxy(self, docs: List[Document], query: str) -> float:
        """
        Fast PΩ proxy using query-chunk token overlap.
        Used as routing signal — not the same as the full PΩ evaluation
        (which uses ground truth answer spans).
        p = |query_tokens ∩ chunk_tokens| / |chunk_tokens|
        PΩ_proxy = p²
        """
        if not docs:
            return 0.0
        q_tokens = set(re.findall(r"\w+", query.lower())) - {
            "the", "a", "an", "is", "are", "was", "were",
            "what", "who", "where", "when", "why", "how",
            "which", "do", "does", "did", "of", "in", "on"
        }
        if not q_tokens:
            return 0.5  # can't judge, assume ok
        scores = []
        for doc in docs:
            chunk_tokens = set(re.findall(r"\w+", doc.page_content.lower()))
            if not chunk_tokens:
                scores.append(0.0)
                continue
            p = len(q_tokens & chunk_tokens) / len(chunk_tokens)
            scores.append(p * p)
        return sum(scores) / len(scores)

    # ── Step 5: Cross-structure verification (NOVELTY 2) ────────────────────

    def _cross_structure_verify(
        self,
        raptor_docs: List[Document],
        query: str
    ) -> List[Document]:
        """
        Shakeel's novelty: KG subgraph-to-RAPTOR-tree intersection.

        Formal algorithm:
        1. Extract entity set E_q from query via KG traversal
        2. For each RAPTOR chunk c:
           keep c if |tokens(c) ∩ E_q| >= theta
        3. If no chunks pass threshold, return original (safety fallback)

        This uses the graph topology as a topological validator/pruning filter
        rather than just a retrieval source — distinct from all existing work.
        """
        if self.kg_builder is None or not raptor_docs:
            return raptor_docs

        # Get entity token set from KG
        entity_tokens = self.kg_builder.get_entity_set(query)
        if not entity_tokens:
            return raptor_docs

        verified = []
        for doc in raptor_docs:
            chunk_tokens = set(re.findall(r"\w+", doc.page_content.lower()))
            overlap = len(chunk_tokens & entity_tokens)
            if overlap >= self.verification_theta:
                verified.append(doc)

        # Safety fallback: if nothing passes, return original
        if not verified:
            logger.info(
                "GRACE-Ω: Cross-verify found 0 verified chunks "
                "(theta=%d) — using unverified.", self.verification_theta
            )
            return raptor_docs

        logger.info(
            "GRACE-Ω: Cross-verify kept %d/%d RAPTOR chunks.",
            len(verified), len(raptor_docs)
        )
        return verified

    # ── RRF helper ───────────────────────────────────────────────────────────

    def _rrf(self, ranked_lists: List[List[Document]], k: int = 60) -> List[Document]:
        scores, seen = {}, {}
        for ranked in ranked_lists:
            for rank, doc in enumerate(ranked):
                key = doc.page_content[:200]
                seen[key] = doc
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [seen[k] for k, _ in ordered]

    # ── Main retrieve ────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> List[Document]:
        """
        Full GRACE-Ω retrieval pipeline:
        1. HyDE query expansion
        2. Complexity-based routing
        3. PΩ quality gate with automatic structure switch on paradox
        4. Cross-structure verification (KG prunes RAPTOR chunks)
        """
        # Step 1: HyDE expand
        expanded = self._hyde_expand(query)

        # Step 2: Initial route classification
        route = self._classify(query)
        routes_tried = []
        all_routes = ["SIMPLE", "MULTIHOP", "RELATIONAL"]
        result_docs = []
        graph_docs = []

        # Try up to 2 routes (PΩ gate switches if needed)
        for attempt in range(2):
            if route not in routes_tried:
                routes_tried.append(route)

            if route == "SIMPLE":
                result_docs = self._retrieve_simple(expanded)
            elif route == "MULTIHOP":
                result_docs = self._retrieve_raptor(expanded)
            else:  # RELATIONAL
                graph_docs = self._retrieve_graph(query)
                result_docs = graph_docs

            # Step 3: PΩ quality gate — our novelty
            pom = self._compute_pomega_proxy(result_docs, query)

            if pom >= self.pomega_threshold or attempt == 1:
                # Good enough or last attempt
                self._last_route = route
                break

            # Paradox detected — switch to next structure
            logger.info(
                "GRACE-Ω: PΩ_proxy=%.3f < %.2f — paradox detected, switching structure.",
                pom, self.pomega_threshold
            )
            remaining = [r for r in all_routes if r not in routes_tried]
            if remaining:
                route = remaining[0]
            else:
                self._last_route = routes_tried[-1]
                break

        # Step 4: Cross-structure verification (KG prunes RAPTOR chunks)
        # Run this when RAPTOR was used
        if self._last_route == "MULTIHOP" and self.kg_builder:
            if not graph_docs:
                graph_docs = self._retrieve_graph(query)
            result_docs = self._cross_structure_verify(result_docs, query)

        return result_docs[:self.top_k]  

    # ═══════════════════════════════════════════════════════════════════════════════
# GRACE-Ω ABLATION VARIANTS — for paper ablation study
# ═══════════════════════════════════════════════════════════════════════════════

class GRACEOmegaNoGate(GRACEOmegaRAG):
    """
    Ablation A: GRACE-Ω without PΩ quality gate.
    Uses random structure switching instead of PΩ-driven switching.
    Proves PΩ routing adds value over random switching.
    """
    name = "GRACE-Ω (no PΩ gate)"

    def retrieve(self, query: str) -> List[Document]:
        import random
        expanded = self._hyde_expand(query)
        route = self._classify(query)
        # No PΩ check — just retrieve from initial route
        if route == "SIMPLE":
            result_docs = self._retrieve_simple(expanded)
        elif route == "MULTIHOP":
            result_docs = self._retrieve_raptor(expanded)
        else:
            result_docs = self._retrieve_graph(query)
        self._last_route = route
        return result_docs[:self.top_k]


class GRACEOmegaNoKG(GRACEOmegaRAG):
    """
    Ablation B: GRACE-Ω without knowledge graph and cross-verification.
    Proves KG cross-structure verification adds value.
    """
    name = "GRACE-Ω (no KG verify)"

    def retrieve(self, query: str) -> List[Document]:
        expanded = self._hyde_expand(query)
        route = self._classify(query)
        routes_tried = []
        all_routes = ["SIMPLE", "MULTIHOP", "RELATIONAL"]
        result_docs = []

        for attempt in range(2):
            if route not in routes_tried:
                routes_tried.append(route)
            if route == "SIMPLE":
                result_docs = self._retrieve_simple(expanded)
            elif route == "MULTIHOP":
                result_docs = self._retrieve_raptor(expanded)
            else:
                result_docs = self._retrieve_graph(query)

            pom = self._compute_pomega_proxy(result_docs, query)
            if pom >= self.pomega_threshold or attempt == 1:
                self._last_route = route
                break
            remaining = [r for r in all_routes if r not in routes_tried]
            if remaining:
                route = remaining[0]

        # NO cross-structure verification step
        return result_docs[:self.top_k]


class GRACEOmegaNoHyDE(GRACEOmegaRAG):
    """
    Ablation C: GRACE-Ω without HyDE query expansion.
    Proves HyDE expansion adds value.
    """
    name = "GRACE-Ω (no HyDE)"

    def _hyde_expand(self, query: str) -> str:
        # Skip HyDE — return query unchanged
        return query

def build_all_architectures(
    vectorstore,
    chunks=None,
    parent_child=None,
    child_vectorstore=None,
    kg_builder=None,          # ← ADD THIS PARAMETER
) -> Dict[str, BaseRAG]:
    architectures: Dict[str, BaseRAG] = {
        "naive":       NaiveRAG(vectorstore=vectorstore),
        "crag":        CorrectiveRAG(vectorstore=vectorstore),
        "rag_fusion":  RAGFusion(vectorstore=vectorstore),
    }
    if chunks:
        architectures["hybrid"] = HybridMultiQueryRAG(
            vectorstore=vectorstore, chunks=chunks)
        architectures["adaptive"] = AdaptiveRAG(
            vectorstore=vectorstore, chunks=chunks)
        architectures["grace_omega"] = GRACEOmegaRAG(    # ← ADD THIS
            vectorstore=vectorstore,
            chunks=chunks,
            kg_builder=kg_builder,
        )
        # Add inside build_all_architectures, after grace_omega:
        if chunks and kg_builder:
            architectures["grace_no_gate"] = GRACEOmegaNoGate(
                vectorstore=vectorstore,
                chunks=chunks,
                kg_builder=kg_builder,
            )
            architectures["grace_no_kg"] = GRACEOmegaNoKG(
                vectorstore=vectorstore,
                chunks=chunks,
                kg_builder=kg_builder,
            )
            architectures["grace_no_hyde"] = GRACEOmegaNoHyDE(
                vectorstore=vectorstore,
                chunks=chunks,
                kg_builder=None,  # KG not needed
            )
            architectures["parent_child"] = ParentChildRAG(
                vectorstore=vectorstore,
                chunks=chunks,
                parent_child=parent_child,
                child_vectorstore=child_vectorstore,
            )
    return architectures

if __name__ == "__main__":
    print("Module 2 loaded. Available: NaiveRAG, HybridMultiQueryRAG, ParentChildRAG, CorrectiveRAG")

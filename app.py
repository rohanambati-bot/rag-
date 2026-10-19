#!/usr/bin/env python3
# app.py — Streamlit Web UI for the RAG Evaluation Framework
"""
Run with:
    streamlit run app.py

Pages:
   Chat            — talk to any RAG architecture (with conversation memory)
   A/B Test        — compare two architectures side-by-side
   Evaluation      — run RAGAS evaluation and see metrics live
   Visualizations  — render all 7 plots from current results
   Cache           — semantic cache statistics + manual lookup
   Settings       — toggle thresholds, view config

Local development only. The API key does not authenticate this UI.
Do not expose it through a tunnel or remote proxy, including one on loopback.
"""

from __future__ import annotations

import sys
import json
import time
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# Suppress noisy warnings from transformers/torchvision
import warnings  # noqa: E402
warnings.filterwarnings("ignore", message=".*torchvision.*")
warnings.filterwarnings("ignore", category=UserWarning)
import os as _os  # noqa: E402
_os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Quiet down 3rd-party loggers in the Streamlit UI
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from security_boundary import session_cache, session_resource, ui_boundary_problems  # noqa: E402

# Use Streamlit's effective settings, not just the checked-in TOML. CLI and env
# overrides must not silently turn this direct-Python UI into a remote service.
# This stops application work; it is not authentication for Streamlit endpoints.
_boundary_problems = ui_boundary_problems(st.get_option, st.context.headers)
if _boundary_problems:
    st.error("UI access refused. " + " ".join(_boundary_problems))
    st.stop()

from configs.config import (  # noqa: E402
    OUTPUTS_DIR, EVAL_RESULTS_PATH, MISTRAL_API_KEY, VECTOR_STORE_DIR,
    FAITHFULNESS_THRESHOLD, CONTEXT_PRECISION_THRESHOLD,
    ANSWER_RELEVANCE_THRESHOLD, NOISE_RATIO_MAX, TOP_K,
    GROQ_API_KEY, GROQ_MODEL,
)

st.set_page_config(
    page_title="RAG Evaluation Framework",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title { font-size: 2.4rem; font-weight: 700;
        background: linear-gradient(90deg, #845ec2 0%, #4ea8de 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem; }
    .subtitle { color: #888; font-size: 1rem; margin-top: 0; }
    .metric-card { background: #f8f9fa; border-radius: 10px; padding: 16px;
        border-left: 4px solid #845ec2; margin: 6px 0; }
    .pass { color: #2e7d32; font-weight: 600; }
    .fail { color: #c62828; font-weight: 600; }
    .arch-pill { display: inline-block; padding: 4px 12px; border-radius: 14px;
        background: #e3f2fd; color: #1565c0; font-size: 0.85rem; margin-right: 6px; }
    .stChatMessage { padding: 0.4rem 0; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# Shared read-only/model resources (Streamlit cache_resource is process-wide)
# ═════════════════════════════════════════════════════════════════════════════

# Every resource comes from serving.CONTEXT, the same stack the API and the CLI use.
# Previously this file bootstrapped its own index from SQuAD while the CLI evaluated
# MS MARCO, and registered only 4 of the 12 architectures — so the UI demonstrated a
# different system from the one the results described.

@st.cache_resource(show_spinner="Loading embedding model…")
def _get_embedding_model():
    from serving import CONTEXT
    return CONTEXT.embedding()


@st.cache_resource(show_spinner="Loading vector store…")
def _get_vectorstore():
    from serving import CONTEXT
    try:
        return CONTEXT.vectorstore()
    except RuntimeError:
        # Fail loudly rather than quietly indexing a different corpus.
        logging.getLogger("app").exception("Vector store unavailable.")
        st.error("Vector store unavailable. Build the configured corpus before serving.")
        st.stop()


@st.cache_resource(show_spinner="Indexing chunks (MS MARCO)…")
def _get_chunks():
    from serving import CONTEXT
    return CONTEXT.chunks()


class LazyArchitectureMap(dict):
    """Dictionary that advertises all architecture display names immediately,
    constructing each architecture lazily on first access.
    """
    def __init__(self):
        super().__init__()
        from serving import CONTEXT
        from module2.rag_architectures import ARCHITECTURE_REGISTRY

        self._context = CONTEXT
        eager = CONTEXT.architectures()
        self._name_to_key: Dict[str, str] = {}
        for key, meta in sorted(ARCHITECTURE_REGISTRY.items()):
            if "kg" not in meta["requires"]:
                display_name = meta.get("name", key)
                self._name_to_key[display_name] = key
                if key in eager:
                    self[display_name] = eager[key]

    def __getitem__(self, name: str):
        if dict.__contains__(self, name):
            return super().__getitem__(name)
        key = self._name_to_key.get(name, name)
        try:
            arch = self._context.architecture(key)
            self[name] = arch
            return arch
        except Exception as exc:
            logging.getLogger("app").error("Failed to build architecture %s: %s", name, exc)
            raise KeyError(name) from exc

    def get(self, name: str, default=None):
        try:
            return self[name]
        except Exception:
            return default

    def __contains__(self, name: object) -> bool:
        return name in self._name_to_key or dict.__contains__(self, name)

    def keys(self):
        return list(self._name_to_key.keys())

    def __iter__(self):
        return iter(list(self._name_to_key.keys()))

    def __len__(self):
        return len(self._name_to_key)

    def items(self):
        for name in list(self._name_to_key.keys()):
            yield (name, self[name])

    def values(self):
        for name in list(self._name_to_key.keys()):
            yield self[name]


@st.cache_resource(show_spinner="Readying RAG architectures…")
def _get_architectures():
    """All registered architectures, keyed by display name for the UI."""
    return LazyArchitectureMap()


def _get_cache(architecture: str = "naive"):
    from phase3.semantic_cache import SemanticCache
    from module2.rag_architectures import ARCHITECTURE_REGISTRY
    key = next((key for key, meta in ARCHITECTURE_REGISTRY.items()
                if meta["name"] == architecture), architecture)
    # Never reuse API/shared-corpus answer caches for private browser queries.
    return session_cache(st.session_state, key, lambda namespace: SemanticCache(
        embedding_model=_get_embedding_model(), namespace=namespace, redis_failure_policy="miss",
    ))


def _get_uploader():
    from module5.document_upload import DocumentUploader
    return session_resource(st.session_state, "uploader", lambda: DocumentUploader(
        embedding_model=_get_embedding_model(),
    ))


# ═════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("###  RAG Framework")
    page = st.radio(
        "Navigate",
        ["Chat", "A/B Test", "Compare All", "Upload Doc",
         "Evaluation", "Failure Analysis",
         "Visualizations", "Cache", "Settings"],
        label_visibility="collapsed",
    )
    st.divider()

    st.markdown("##### Status")
    if GROQ_API_KEY:
        api_state = "Groq (Llama 3.1)"
    elif MISTRAL_API_KEY:
        api_state = "Mistral"
    else:
        api_state = "Extractive fallback"
    st.caption(f"LLM: {api_state}")
    st.caption(f"Vector Store: {'Ready' if Path(VECTOR_STORE_DIR).exists() else ' Not built'}")

    st.divider()
    st.caption("Local-only UI · no user authentication · do not tunnel/proxy")
    if st.button("Clear my answer caches", use_container_width=True):
        receipts = [cache.clear() for cache in st.session_state.get("_private_answer_caches", {}).values()]
        if any(receipt is False for receipt in receipts):
            st.warning("Local caches cleared; shared invalidation could not be confirmed. Retry when Redis recovers.")
        else:
            st.success("Your answer caches cleared.")
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# Page: Chat
# ═════════════════════════════════════════════════════════════════════════════

def page_chat():
    st.markdown('<div class="main-title"> RAG Chat</div>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Multi-turn conversations with retrieval-augmented answers.</p>',
                unsafe_allow_html=True)

    arch_names = list(_get_architectures().keys())
    arch_choice = st.selectbox(
        "Architecture",
        arch_names,
        index=min(3, len(arch_names) - 1),
    )
    use_cache = st.toggle("Use semantic cache", value=True,
                           help="Skip retrieval if a similar query is cached.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    col_clear, _ = st.columns([1, 5])
    with col_clear:
        if st.button("New conversation"):
            st.session_state.chat_history = []
            st.rerun()

    # Render history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"Retrieved context ({len(msg['sources'])} chunks)"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(f"**Chunk {i}:** {src[:400]}…" if len(src) > 400 else
                                    f"**Chunk {i}:** {src}")
            if "meta" in msg:
                st.caption(msg["meta"])

    query = st.chat_input("Ask a question…")
    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("_Thinking…_")
            try:
                cache = _get_cache(arch_choice) if use_cache else None
                cached = cache.lookup(query) if cache else None

                if cached:
                    answer = cached["answer"]
                    sources = cached.get("contexts", [])
                    meta = (f" Cache hit (similarity {cached['similarity_score']:.3f}) · "
                            f"latency 0 ms")
                else:
                    rag = _get_architectures()[arch_choice]
                    # Use only the current query for retrieval (no history mixing)
                    result = rag.run(query)
                    answer = result["answer"]
                    sources = result.get("contexts", [])

                    # Confidence scoring
                    from module5.confidence import ConfidenceScorer
                    scorer = ConfidenceScorer()
                    conf = scorer.score(query=query, answer=answer, contexts=sources)
                    conf_str = f" · {conf['emoji']} {conf['score']:.0f}% confidence"

                    meta = (f" {arch_choice} · latency {result.get('latency_ms', 0):.0f} ms · "
                            f"{len(sources)} chunks{conf_str}")
                    if cache:
                        store_with_citations = getattr(cache, "store_with_citations", None)
                        if store_with_citations is not None:
                            store_with_citations(
                                query, answer, sources, result.get("citations", [])
                            )
                        else:
                            cache.store(query, answer, sources)

                placeholder.markdown(answer)
                if sources:
                    with st.expander(f" Retrieved context ({len(sources)} chunks)"):
                        for i, src in enumerate(sources, 1):
                            display = src if len(src) <= 400 else src[:400] + "…"
                            st.markdown(f"**Chunk {i}:** {display}")
                st.caption(meta)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "meta": meta,
                })
            except Exception:
                logging.getLogger("app").exception("Chat request failed.")
                placeholder.error("Request failed; check the local server logs.")
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": "Request failed; check the local server logs."}
                )


# ═════════════════════════════════════════════════════════════════════════════
# Page: A/B Test
# ═════════════════════════════════════════════════════════════════════════════

def page_ab():
    st.markdown('<div class="main-title"> A/B Architecture Test</div>',
                unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Run the same query through two architectures and compare.</p>',
                unsafe_allow_html=True)

    arch_options = list(_get_architectures().keys())
    c1, c2 = st.columns(2)
    with c1:
        arch_a = st.selectbox("Variant A", arch_options, index=0, key="ab_a")
    with c2:
        arch_b = st.selectbox("Variant B", arch_options, index=3, key="ab_b")

    query = st.text_input(
        "Question",
        placeholder="e.g. What was the main cause of the French Revolution?",
    )
    run = st.button("Run comparison", type="primary", disabled=not query)

    if run and query:
        archs = _get_architectures()
        with st.spinner("Running both architectures…"):
            res_a = archs[arch_a].run(query)
            res_b = archs[arch_b].run(query)

        col_a, col_b = st.columns(2)
        for col, label, res in [(col_a, arch_a, res_a), (col_b, arch_b, res_b)]:
            with col:
                st.markdown(f"### {label}")
                st.markdown(
                    f'<div class="metric-card"><b>Answer:</b><br>{res["answer"]}</div>',
                    unsafe_allow_html=True,
                )
                m1, m2 = st.columns(2)
                m1.metric("Latency", f"{res['latency_ms']:.0f} ms")
                m2.metric("Chunks", len(res.get("contexts", [])))
                with st.expander("Retrieved context"):
                    for i, c in enumerate(res.get("contexts", []), 1):
                        st.markdown(f"**{i}.** {c[:300]}…" if len(c) > 300 else f"**{i}.** {c}")

        # Diff summary
        st.divider()
        st.markdown("#### Comparison")
        delta = res_b["latency_ms"] - res_a["latency_ms"]
        chunks_a = len(res_a.get("contexts", []))
        chunks_b = len(res_b.get("contexts", []))
        st.markdown(
            f"- **Latency Δ**: {delta:+.0f} ms ({arch_b} vs {arch_a})\n"
            f"- **Chunks retrieved**: A={chunks_a}, B={chunks_b}\n"
            f"- **Answer length**: A={len(res_a['answer'])} chars, B={len(res_b['answer'])} chars"
        )


# ═════════════════════════════════════════════════════════════════════════════
# Page: Evaluation
# ═════════════════════════════════════════════════════════════════════════════

def page_eval():
    st.markdown('<div class="main-title">Evaluation</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Run a quick RAGAS-style benchmark over synthetic queries.</p>',
        unsafe_allow_html=True,
    )

    arch_options = list(_get_architectures().keys())
    selected = st.multiselect(
        "Architectures to evaluate", arch_options, default=arch_options,
    )
    samples = st.slider("Number of synthetic queries", 5, 100, 15)
    use_ragas = st.toggle(
        "Use RAGAS LLM-judged metrics (requires a Groq or Mistral key)",
        value=bool(GROQ_API_KEY or MISTRAL_API_KEY),
        disabled=not (GROQ_API_KEY or MISTRAL_API_KEY),
    )

    if not selected:
        st.warning("Select at least one architecture.")
        return

    if st.button("Run evaluation", type="primary"):
        from module3.evaluation import EvaluationPipeline, RAGASEvaluator
        synthetic_path = Path(__file__).parent / "data" / "synthetic_dataset.json"

        # Load or synthesise queries
        if synthetic_path.exists():
            synthetic = json.loads(synthetic_path.read_text())[:samples]
        else:
            synthetic = _bootstrap_synthetic(samples)

        ragas = RAGASEvaluator() if use_ragas else None
        pipeline = EvaluationPipeline(ragas_evaluator=ragas)
        archs = _get_architectures()

        progress = st.progress(0.0)
        status = st.empty()
        results: Dict[str, pd.DataFrame] = {}
        for i, name in enumerate(selected, 1):
            status.markdown(f"Evaluating **{name}** ({i}/{len(selected)})…")
            df = pipeline.evaluate_architecture(
                architecture_name=name,
                rag_system=archs[name],
                synthetic_dataset=synthetic,
                max_samples=samples,
            )
            results[name] = df
            progress.progress(i / len(selected))

        status.empty()
        progress.empty()

        summary = pipeline.compare_architectures(results)
        st.success(f"Evaluated {samples} queries × {len(selected)} architectures.")

        st.markdown("#### Summary")
        st.dataframe(summary, use_container_width=True)

        # KPI cards for the best architecture
        if "faithfulness" in summary.columns and not summary.empty:
            best = summary.loc[summary["faithfulness"].idxmax()]
            st.markdown(f" **Best on Faithfulness**: `{best['architecture']}` — "
                        f"{best['faithfulness']:.3f}")

        # Quality gate badges
        st.markdown("#### Quality gate per architecture")
        for _, row in summary.iterrows():
            cols = st.columns(5)
            cols[0].markdown(f"**{row['architecture']}**")
            for i, (metric, threshold, higher) in enumerate([
                ("faithfulness", FAITHFULNESS_THRESHOLD, True),
                ("context_precision", CONTEXT_PRECISION_THRESHOLD, True),
                ("answer_relevance", ANSWER_RELEVANCE_THRESHOLD, True),
                ("noise_ratio", NOISE_RATIO_MAX, False),
            ]):
                val = row.get(metric)
                if val is None or pd.isna(val):
                    cols[i + 1].markdown(f"{metric}: —")
                    continue
                ok = (val >= threshold) if higher else (val <= threshold)
                cls = "pass" if ok else "fail"
                emoji = "✓" if ok else "✗"
                cols[i + 1].markdown(
                    f"<span class='{cls}'>{emoji} {metric}: {val:.3f}</span>",
                    unsafe_allow_html=True,
                )

        # Persist
        out_csv = Path(EVAL_RESULTS_PATH)
        if out_csv.exists():
            st.caption(f"Saved to `{out_csv.relative_to(Path(__file__).parent)}`")

        # Download
        csv_bytes = summary.to_csv(index=False).encode()
        st.download_button("⬇  Download CSV", csv_bytes, "evaluation_summary.csv", "text/csv")
        md = _summary_markdown(summary)
        st.download_button("⬇  Download Markdown report", md.encode(),
                           "evaluation_report.md", "text/markdown")


def _bootstrap_synthetic(n: int) -> List[Dict[str, Any]]:
    """Tiny fallback dataset when synthetic_dataset.json doesn't exist yet."""
    samples = [
        {"question": "What is the capital of France?",
         "ground_truth": "Paris", "query_type": "simple"},
        {"question": "Who wrote Romeo and Juliet?",
         "ground_truth": "William Shakespeare", "query_type": "simple"},
        {"question": "What year did World War II end?",
         "ground_truth": "1945", "query_type": "simple"},
        {"question": "What is the largest planet in the solar system?",
         "ground_truth": "Jupiter", "query_type": "simple"},
        {"question": "Who painted the Mona Lisa?",
         "ground_truth": "Leonardo da Vinci", "query_type": "simple"},
    ]
    out = []
    while len(out) < n:
        out.extend(samples)
    return out[:n]


def _summary_markdown(summary_df: pd.DataFrame) -> str:
    lines = [
        "# RAG Evaluation Report",
        "",
        f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
        "## Architecture Comparison",
        "",
        summary_df.to_markdown(index=False),
        "",
    ]
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# Page: Visualizations
# ═════════════════════════════════════════════════════════════════════════════

def page_viz():
    st.markdown('<div class="main-title"> Visualizations</div>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Plots from the latest evaluation.</p>',
                unsafe_allow_html=True)

    plot_dir = Path(OUTPUTS_DIR) / "plots"
    has_data = Path(EVAL_RESULTS_PATH).exists()
    has_plots = plot_dir.exists() and any(plot_dir.glob("*.png"))

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button(" Regenerate from current results", disabled=not has_data):
            from module4.visualization import generate_all_plots
            df = pd.read_csv(EVAL_RESULTS_PATH)
            raw = Path(OUTPUTS_DIR) / "raw_eval_results.csv"
            raw_df = pd.read_csv(raw) if raw.exists() else None
            with st.spinner("Generating plots…"):
                generate_all_plots(df, raw_df)
            st.success("Plots regenerated.")
            st.rerun()
    with c2:
        if st.button(" Generate demo plots (benchmark data)"):
            from module4.visualization import demo_plots
            with st.spinner("Generating demo plots…"):
                demo_plots()
            st.success("Demo plots generated.")
            st.rerun()

    if not has_plots:
        st.info("No plots yet — click one of the buttons above.")
        return

    pngs = sorted(plot_dir.glob("*.png"))
    cols = st.columns(2)
    for i, png in enumerate(pngs):
        with cols[i % 2]:
            st.image(str(png), caption=png.stem.replace("_", " ").title(),
                     use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# Page: Cache
# ═════════════════════════════════════════════════════════════════════════════

def page_cache():
    st.markdown('<div class="main-title"> Semantic Cache</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Session-isolated vector cache (Redis with in-memory fallback).</p>',
        unsafe_allow_html=True,
    )

    cache = _get_cache()
    stats = cache.stats()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total queries", stats.get("total_queries", 0))
    c2.metric("Hits", stats.get("hits", 0))
    c3.metric("Misses", stats.get("misses", 0))
    c4.metric("Hit rate", f"{stats.get('hit_rate', 0) * 100:.1f}%")

    st.divider()
    st.markdown("#### Try a lookup")
    q = st.text_input("Query", placeholder="Enter a query to test cache lookup")
    if st.button("Lookup") and q:
        with st.spinner("Searching cache…"):
            res = cache.lookup(q)
        if res:
            st.success(f"HIT — similarity {res['similarity_score']:.3f}")
            st.markdown(f"**Cached answer:** {res['answer']}")
        else:
            st.info("MISS — no semantically similar entry above threshold.")

    st.divider()
    if st.button(" Clear cache", type="secondary"):
        if cache.clear() is False:
            st.warning("Local cache cleared; shared invalidation could not be confirmed. Retry when Redis recovers.")
        else:
            st.success("Cache cleared.")
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# Page: Settings
# ═════════════════════════════════════════════════════════════════════════════

def page_settings():
    st.markdown('<div class="main-title"> Settings</div>', unsafe_allow_html=True)

    st.markdown("#### Quality Gate Thresholds")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Faithfulness", f"≥ {FAITHFULNESS_THRESHOLD}")
        st.metric("Context Precision", f"≥ {CONTEXT_PRECISION_THRESHOLD}")
    with c2:
        st.metric("Answer Relevance", f"≥ {ANSWER_RELEVANCE_THRESHOLD}")
        st.metric("Noise Ratio", f"≤ {NOISE_RATIO_MAX}")

    st.divider()
    st.markdown("#### Configuration")
    cfg = {
        "GROQ_API_KEY set": bool(GROQ_API_KEY),
        "GROQ_MODEL": GROQ_MODEL,
        "MISTRAL_API_KEY set": bool(MISTRAL_API_KEY),
        "TOP_K": TOP_K,
        "Vector store": VECTOR_STORE_DIR,
        "Outputs": OUTPUTS_DIR,
    }
    st.json(cfg)

    st.divider()
    st.markdown("#### About")
    try:
        n_archs = len(_get_architectures())
    except Exception:
        n_archs = "12"
    st.markdown(
        f"Production-grade RAG evaluation framework — Capstone Phase 3. "
        f"{n_archs} architectures · RAGAS metrics · semantic caching · CI/CD quality gates."
    )


# ═════════════════════════════════════════════════════════════════════════════
# Page: Document Upload
# ═════════════════════════════════════════════════════════════════════════════

def page_upload():
    st.markdown('<div class="main-title"> Document Upload</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Upload your own PDF/TXT/DOCX and query it with any architecture.</p>',
        unsafe_allow_html=True,
    )

    from module5.confidence import ConfidenceScorer

    uploader = _get_uploader()

    uploaded = st.file_uploader(
        "Upload a document",
        type=["txt", "pdf", "md", "csv", "docx"],
        help="Supported: .txt, .pdf, .md, .csv, .docx",
    )

    if uploaded:
        import hashlib
        content = uploaded.getvalue()
        upload_digest = hashlib.sha256(content).hexdigest()
        if (not uploader.is_ready or uploader.filename != uploaded.name
                or st.session_state.get("_upload_digest") != upload_digest):
            with st.spinner(f"Indexing {uploaded.name}…"):
                try:
                    n = uploader.index_documents(content, uploaded.name)
                    st.session_state["_upload_digest"] = upload_digest
                    st.success(f" Indexed **{uploaded.name}** → {n} chunks")
                except Exception:
                    logging.getLogger("app").exception("Document upload failed.")
                    st.error("Upload failed. Check the file format/limits and local server logs.")

    if uploader.is_ready:
        st.info(f" Active document: **{uploader.filename}** ({uploader.num_chunks} chunks)")

        # Uploaded documents have their own temporary corpus, so only expose
        # architectures whose constructors can consume that vector store directly.
        # Previously the UI listed every global architecture but silently fell back to
        # Naive RAG for most selections, making the displayed experiment misleading.
        upload_arch_options = ["Naive RAG", "CRAG", "Rerank RAG"]
        arch_choice = st.selectbox(
            "Architecture", upload_arch_options, index=0, key="upload_arch"
        )
        query = st.text_input("Ask a question about your document", key="upload_q")

        if st.button("Search", disabled=not query) and query:
            with st.spinner("Searching…"):
                try:
                    from module2.rag_architectures import (
                        CorrectiveRAG, NaiveRAG, RerankRAG,
                    )
                    if arch_choice == "CRAG":
                        rag = CorrectiveRAG(vectorstore=uploader.vectorstore)
                    elif arch_choice == "Rerank RAG":
                        rag = RerankRAG(
                            vectorstore=uploader.vectorstore,
                            chunks=uploader.chunks,
                        )
                    else:
                        rag = NaiveRAG(vectorstore=uploader.vectorstore)
                    result = rag.run(query)

                    docs = result.get("context_docs", [])
                    if not docs:
                        st.warning("No matching passages found in the uploaded document.")
                        return

                    answer = result["answer"]

                    # Confidence
                    scorer = ConfidenceScorer()
                    contexts = result.get("contexts", [])
                    conf = scorer.score(query=query, answer=answer, contexts=contexts)

                    # Display
                    st.markdown("### Answer")
                    st.markdown(answer)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Confidence", f"{conf['score']:.0f}%")
                    c2.markdown(f"**{conf['emoji']} {conf['label']}**")
                    c3.metric("Chunks used", len(docs))

                    with st.expander(" Retrieved chunks"):
                        for i, doc in enumerate(docs, 1):
                            st.markdown(f"**Chunk {i}:** {doc.page_content[:300]}")
                except Exception:
                    logging.getLogger("app").exception("Uploaded document search failed.")
                    st.error("Search failed; check the local server logs.")

        if st.button("Clear document"):
            uploader.clear()
            st.session_state.pop("_upload_digest", None)
            st.rerun()
    else:
        st.info("Upload a document above to get started.")


# ═════════════════════════════════════════════════════════════════════════════
# Page: Architecture Comparison
# ═════════════════════════════════════════════════════════════════════════════

def page_compare():
    st.markdown('<div class="main-title"> Architecture Comparison</div>',
                unsafe_allow_html=True)
    n_archs = len(_get_architectures())
    st.markdown(
        f'<p class="subtitle">Run all {n_archs} architectures on the same query. See who wins.</p>',
        unsafe_allow_html=True,
    )

    from module5.comparison import ArchitectureComparator
    from module5.confidence import ConfidenceScorer

    query = st.text_input(
        "Question",
        placeholder="e.g. what is a corporation",
        key="compare_q",
    )

    if st.button(" Compare all architectures", type="primary", disabled=not query) and query:
        archs = _get_architectures()
        comparator = ArchitectureComparator(archs)
        scorer = ConfidenceScorer()

        with st.spinner(f"Running all {len(archs)} architectures…"):
            results = comparator.compare_with_confidence(query, scorer)

        # Rank by confidence
        ranked = comparator.rank_by_confidence(results)

        # Winner banner
        winner = ranked[0]
        st.success(
            f"**Winner: {winner.architecture}** — "
            f"Confidence {winner.confidence:.0f}% · {winner.latency_ms:.0f}ms"
        )

        # Comparison table
        st.markdown("#### Results")
        for i, r in enumerate(ranked):
            medals = ["🥇", "🥈", "🥉"] + [f"{j+4}️⃣" for j in range(len(ranked))]
            medal = medals[i] if i < len(medals) else f"#{i+1}"
            with st.container():
                cols = st.columns([0.5, 2, 1, 1, 1])
                cols[0].markdown(f"### {medal}")
                cols[1].markdown(f"**{r.architecture}**")
                cols[2].metric("Confidence", f"{r.confidence:.0f}%")
                cols[3].metric("Latency", f"{r.latency_ms:.0f}ms")
                cols[4].metric("Chunks", r.num_chunks)

                st.markdown(f"> {r.answer[:300]}{'…' if len(r.answer) > 300 else ''}")
                st.divider()

        # Latency comparison bar
        st.markdown("#### Latency Comparison")
        latency_data = pd.DataFrame([
            {"Architecture": r.architecture, "Latency (ms)": r.latency_ms}
            for r in results
        ])
        st.bar_chart(latency_data.set_index("Architecture"))


# ═════════════════════════════════════════════════════════════════════════════
# Page: Failure Analysis
# ═════════════════════════════════════════════════════════════════════════════

def page_failure():
    st.markdown('<div class="main-title">🔍 Failure Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Diagnose WHY answers fail — maps to 7 RAG failure points (Barnett et al.).</p>',
        unsafe_allow_html=True,
    )

    from module5.failure_analysis import FailureAnalyzer, FAILURE_POINTS

    # Show the 7 failure points reference
    with st.expander("The 7 RAG Failure Points (Barnett et al., IEEE/ACM CAIN 2024)"):
        for fp in FAILURE_POINTS:
            st.markdown(f"**{fp.id}: {fp.name}** — {fp.description}")
            st.caption(f"Solution: {fp.solution}")

    st.divider()

    mode = st.radio("Mode", ["Single query analysis", "Batch analysis (synthetic dataset)"],
                    horizontal=True)

    analyzer = FailureAnalyzer()

    if mode == "Single query analysis":
        archs = _get_architectures()
        arch_name = st.selectbox("Architecture", list(archs.keys()), key="fa_arch")
        query = st.text_input("Question", key="fa_q",
                              placeholder="e.g. what is a corporation")
        ground_truth = st.text_input("Ground truth answer (optional)", key="fa_gt",
                                     placeholder="e.g. A corporation is a company...")

        if st.button("Analyze", disabled=not query) and query:
            rag = archs[arch_name]

            with st.spinner("Running RAG + analysis…"):
                result = rag.run(query)
                analysis = analyzer.analyze(
                    query=query,
                    answer=result["answer"],
                    contexts=result.get("contexts", []),
                    ground_truth=ground_truth or None,
                )

            # Quality score
            score = analysis["quality_score"]
            color = "pass" if score >= 70 else "fail"
            st.markdown(
                f"### Quality Score: <span class='{color}'>{score}/100</span>",
                unsafe_allow_html=True,
            )

            st.markdown(f"**Answer:** {result['answer']}")

            # Detected failures
            if analysis["has_failure"]:
                st.error(f" {len(analysis['detected_failures'])} failure point(s) detected:")
                for fp in analysis["detected_failures"]:
                    st.markdown(
                        f"- **{fp.id}: {fp.name}** — {fp.evidence}\n"
                        f"  -  *{fp.solution}*"
                    )
            else:
                st.success(" No failure points detected — answer looks good!")

            # Recommendations
            if analysis["recommendations"]:
                st.markdown("#### Recommendations")
                for rec in analysis["recommendations"]:
                    st.markdown(f"- {rec}")

    else:
        # Batch analysis
        st.markdown("Run failure analysis on the synthetic evaluation dataset.")
        samples = st.slider("Samples to analyze", 10, 100, 30, key="fa_samples")
        archs = _get_architectures()
        arch_name = st.selectbox(
            "Architecture for batch analysis",
            list(archs.keys()),
            index=0,
            key="fa_batch_arch",
        )

        if st.button("Run batch analysis"):
            syn_path = Path(__file__).parent / "data" / "synthetic_dataset.json"
            if not syn_path.exists():
                st.error("No synthetic dataset found. Run evaluation first.")
                return

            synthetic = json.loads(syn_path.read_text())[:samples]
            rag = archs[arch_name]

            progress = st.progress(0.0)
            results_for_analysis = []

            for i, sample in enumerate(synthetic):
                try:
                    result = rag.run(sample["question"])
                    results_for_analysis.append({
                        "query": sample["question"],
                        "answer": result["answer"],
                        "contexts": result.get("contexts", []),
                        "ground_truth": sample.get("ground_truth"),
                    })
                except Exception:
                    pass
                progress.progress((i + 1) / samples)

            progress.empty()

            batch_result = analyzer.batch_analyze(results_for_analysis)

            # Display
            st.markdown("#### Batch Results")
            c1, c2, c3 = st.columns(3)
            c1.metric("Avg Quality", f"{batch_result['avg_quality']:.0f}/100")
            c2.metric("Failure Rate", f"{batch_result['failure_rate']*100:.0f}%")
            c3.metric("Most Common", batch_result["most_common_failure"] or "None")

            # Failure distribution chart
            st.markdown("#### Failure Point Distribution")
            fp_df = pd.DataFrame([
                {"Failure Point": k, "Count": v}
                for k, v in batch_result["failure_distribution"].items()
                if v > 0
            ])
            if not fp_df.empty:
                st.bar_chart(fp_df.set_index("Failure Point"))
            else:
                st.success("No failures detected in the batch!")


# ═════════════════════════════════════════════════════════════════════════════
# Router
# ═════════════════════════════════════════════════════════════════════════════

PAGE_MAP = {
    "Chat":            page_chat,
    "A/B Test":        page_ab,
    "Compare All":     page_compare,
    "Upload Doc":      page_upload,
    "Evaluation":      page_eval,
    "Failure Analysis": page_failure,
    "Visualizations":  page_viz,
    "Cache":           page_cache,
    "Settings":        page_settings,
}
PAGE_MAP[page]()

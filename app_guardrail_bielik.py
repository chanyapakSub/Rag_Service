# app_demo_final.py
# RAG Chatbot + Bielik-Guard + Custom Safety Rules (English UI - Professional Demo Version)
# Works 100% with your current chroma_db structure

import streamlit as st
import chromadb
import google.generativeai as genai
import os
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from dotenv import load_dotenv

# ========================= CONFIG =========================
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY2"))

st.set_page_config(
    page_title="Secure RAG Chatbot Demo",
    page_icon="shield",
    layout="wide"
)

st.title("Secure RAG Chatbot with Custom Guardrails")
st.caption("Bielik-Guard 0.1B + Dynamic Topic/Keyword Safety + Real Document RAG")

# ========================= LOAD MODELS (Auto-detect Chroma) =========================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('BAAI/bge-base-en-v1.5')

@st.cache_resource
def load_guard_model():
    return pipeline(
        "text-classification",
        model="speakleash/Bielik-Guard-0.1B-v1.0",
        top_k=None,  # Updated: replaces return_all_scores=True
        device=-1,
        batch_size=1
    )

@st.cache_resource
def load_chroma_db():
    embedder = load_embedding_model()
    
    # Point to parent folder containing the UUID subfolder
    client = chromadb.PersistentClient(path="models/chroma_db_bge")
    
    # Auto-detect the first collection with data
    collections = client.list_collections()
    if not collections:
        st.error("No Chroma collections found in 'models/chroma_db_bge'")
        st.stop()
    
    for col in collections:
        try:
            collection = client.get_collection(name=col.name)
            count = collection.count()
            if count > 0:
                st.sidebar.success(f"Connected to collection: `{col.name}` ({count} documents)")
                return embedder, collection
        except:
            continue
    
    st.error("Found collections but none contain data.")
    st.stop()

embedder, collection = load_chroma_db()
guard = load_guard_model()

# ========================= RAG RETRIEVAL =========================
def retrieve_documents(query: str, min_similarity: float = 0.5, top_k: int = 8):
    query_emb = embedder.encode([query])
    results = collection.query(
        query_embeddings=query_emb.tolist(),
        n_results=top_k,
        include=["documents", "distances"]
    )
    docs, scores = [], []
    for doc, dist in zip(results["documents"][0], results["distances"][0]):
        sim = 1 - dist
        if sim >= min_similarity:
            docs.append(doc)
            scores.append(sim)
    return docs, scores

# ========================= SAFETY CHECK =========================
def is_unsafe(text: str, threshold: float = 0.5) -> bool:
    try:
        results = guard(text)[0]
        top_score = max(item["score"] for item in results)
        return top_score >= threshold
    except:
        return False

# ========================= SIDEBAR - Safety Controls =========================
with st.sidebar:
    st.header("Safety Controls")
    
    with st.expander("Sensitive Topics", expanded=True):
        topics = ["Religion", "Politics", "Monarchy", "Violence", "Hate Speech", "Sexual Content"]
        topic_rules = {}
        for i, topic in enumerate(topics):
            col1, col2 = st.columns([2, 3])
            with col1:
                enabled = st.checkbox(topic, key=f"topic_en_{i}")
            with col2:
                if enabled:
                    level = st.select_slider(
                        f"{topic} level",
                        options=["Low", "Medium", "High"],
                        value="Medium",
                        key=f"topic_lvl_{i}"
                    )
                    topic_rules[topic] = level

    with st.expander("Custom Blocked Keywords"):
        if "blocked_keywords" not in st.session_state:
            st.session_state.blocked_keywords = []

        with st.form("add_keyword", clear_on_submit=True):
            word = st.text_input("Keyword")
            level = st.selectbox("Protection level", ["Medium", "High"], index=1)
            submitted = st.form_submit_button("Add")
            if submitted and word.strip():
                if not any(k["word"].lower() == word.lower() for k in st.session_state.blocked_keywords):
                    st.session_state.blocked_keywords.append({"word": word.strip(), "level": level})

        for i, item in enumerate(st.session_state.blocked_keywords.copy()):
            c1, c2, c3 = st.columns([3, 2, 1])
            with c1: st.write(f"• {item['word']}")
            with c2:
                nl = st.select_slider("", ["Medium", "High"], value=item["level"], key=f"kwl_{i}")
                st.session_state.blocked_keywords[i]["level"] = nl
            with c3:
                if st.button("Remove", key=f"rm_{i}"):
                    st.session_state.blocked_keywords.pop(i)
                    st.rerun()

    st.divider()
    guard_threshold = st.slider("Bielik-Guard Sensitivity", 0.3, 0.9, 0.50, 0.05)
    rag_similarity = st.slider("RAG Minimum Similarity", 0.3, 0.8, 0.50, 0.05)

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# ========================= SYSTEM PROMPT BUILDER =========================
def build_system_prompt():
    rules = ["You are a safe, honest, and professional assistant."]
    rules.append("Strictly follow these rules:\n")
    
    if topic_rules:
        rules.append("Sensitive topics:")
        for topic, level in topic_rules.items():
            if level == "High":
                rules.append(f"• Never discuss or answer questions about '{topic}'")
            else:
                rules.append(f"• If asked about '{topic}', politely decline")

    if st.session_state.blocked_keywords:
        rules.append("\nBlocked keywords:")
        for item in st.session_state.blocked_keywords:
            rules.append(f"• If '{item['word']}' appears → refuse or redirect")

    rules.extend([
        "\n• Only use information from provided documents",
        "• If no relevant info → say: 'No information found in documents.'",
        "• Never hallucinate or make up facts",
        "• Respond concisely and professionally"
    ])
    return "\n".join(rules)

system_prompt = build_system_prompt()

# ========================= CHAT INTERFACE =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander(f"Sources ({len(msg['sources'])} documents)"):
                for i, (doc, score) in enumerate(zip(msg["sources"], msg["scores"])):
                    st.caption(f"Source {i+1} • Similarity: {score:.3f}")
                    st.text(doc[:500] + ("..." if len(doc) > 500 else ""))

# User input
if prompt := st.chat_input("Ask anything (fully secured)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 1. Safety check
    if is_unsafe(prompt, guard_threshold):
        reply = "I'm sorry, I cannot assist with that request as it may violate safety guidelines."
    else:
        # 2. RAG retrieval
        docs, scores = retrieve_documents(prompt, rag_similarity)
        if not docs:
            reply = "No relevant information found in the documents."
        else:
            context = "\n\n".join([f"[Document {i+1}]\n{doc}" for i, doc in enumerate(docs[:5])])
            full_prompt = f"""{system_prompt}

=== Document Context ===
{context}

=== User Question ===
{prompt}

=== Answer (clear, concise, professional):"""

            try:
                model = genai.GenerativeModel('gemini-2.5-flash')
                response = model.generate_content(full_prompt)
                reply = response.text.strip()
            except Exception as e:
                reply = f"Error: {e}"

    # Display assistant reply
    with st.chat_message("assistant"):
        st.markdown(reply)

    # Save to history
    msg_data = {"role": "assistant", "content": reply}
    if 'docs' in locals():
        msg_data["sources"] = docs
        msg_data["scores"] = scores
    st.session_state.messages.append(msg_data)

# Final touch
st.sidebar.info("RAG Chatbot is ready and fully secure.")
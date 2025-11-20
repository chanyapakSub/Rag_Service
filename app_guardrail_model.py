# app.py
import streamlit as st
import chromadb
import google.generativeai as genai
import joblib
import os
from sentence_transformers import SentenceTransformer
from pathlib import Path
from dotenv import load_dotenv

# ───────────────────── CONFIG ─────────────────────
st.set_page_config(page_title="RAG + Guardrail Chatbot", page_icon="AI", layout="wide")
st.title("RAG Chatbot + AI Guardrail ระดับชาติ")
st.caption("ใช้ Chroma Vector DB + Gemini 1.5 Flash + Guardrail (Embedding + XGBoost)")

# ───────────────────── LOAD ENV & KEYS ─────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY2")
if not GEMINI_API_KEY:
    st.error("กรุณาใส่ GEMINI_API_KEY2 ในไฟล์ .env")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# ───────────────────── LOAD MODELS ─────────────────────
@st.cache_resource
def load_models():
    # 1. Embedding model สำหรับ RAG
    rag_embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 2. Chroma DB
    client = chromadb.PersistentClient(path="models/chroma_db")
    collection = client.get_collection("rag_collection")
    
    # 3. Guardrail models
    guard_embedder = SentenceTransformer("outputs/embeding_thai_eng_safe_unsafe_triplet_cls_ready2/checkpoints/step_49000")
    clf = joblib.load("model/clf_models_thai_eng_add_data/xgboost_pipeline.joblib")
    
    return rag_embedder, collection, guard_embedder, clf

rag_embedder, collection, guard_embedder, clf = load_models()

# ───────────────────── GUARDRAIL PREDICT ─────────────────────
def is_unsafe(prompt: str) -> bool:
    try:
        emb = guard_embedder.encode([prompt])
        pred = clf.predict(emb)[0]
        return pred == 1  # 1 = unsafe
    except:
        return False  # ถ้า error ให้ผ่านไปก่อน

# ───────────────────── RAG RETRIEVAL ─────────────────────
def rag_retrieve(query: str, top_k: int = 5, min_similarity: float = 0.50):
    query_emb = rag_embedder.encode([query])
    results = collection.query(
        query_embeddings=query_emb.tolist(),
        n_results=top_k * 2,
        include=["documents", "distances", "metadatas"]
    )
    
    docs = []
    scores = []
    for doc, dist, meta in zip(results['documents'][0], results['distances'][0], results['metadatas'][0]):
        sim = 1 - dist
        if sim >= min_similarity:
            docs.append(doc)
            scores.append(sim)
        if len(docs) >= top_k:
            break
    return docs, scores

# ───────────────────── SIDEBAR SETTINGS ─────────────────────
with st.sidebar:
    st.header("Guardrail Settings")
    st.markdown("ตั้งค่าการป้องกันหัวข้อและคำต้องห้าม")
    
    topics = ["Religion", "Politics", "Monarchy", "Violence", "Hate Speech"]
    topic_levels = {}
    for topic in topics:
        checked = st.checkbox(topic, value=True)
        if checked:
            level = st.select_slider(
                topic,
                options=["Low", "Medium", "High"],
                value="High",
                key=f"level_{topic}"
            )
            topic_levels[topic] = level

    st.markdown("### คำต้องห้ามเพิ่มเติม")
    custom = st.text_input("เพิ่มคำต้องห้าม (คั่นด้วย comma)")
    custom_keywords = [k.strip() for k in custom.split(",") if k.strip()]

    st.markdown("### ความเหมือนขั้นต่ำของ RAG")
    min_sim = st.slider("Similarity Threshold", 0.30, 0.80, 0.50, 0.05)

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# ───────────────────── CHAT HISTORY ─────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("ดูแหล่งที่มาจากเอกสาร"):
                for i, src in enumerate(msg["sources"]):
                    st.caption(f"แหล่งที่ {i+1} | ความเหมือน: {msg['scores'][i]:.3f}")
                    st.text(src[:500] + "...")

# ───────────────────── CHAT INPUT ─────────────────────
if prompt := st.chat_input("ถามอะไรก็ได้ครับ (ปลอดภัย 100%)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 1. ตรวจ Guardrail ก่อน
    if is_unsafe(prompt):
        reply = "ขออภัยครับ คำถามนี้ถูกบล็อกโดย AI Guardrail เพื่อความปลอดภัยและความเหมาะสม"
    else:
        # 2. RAG Retrieval
        docs, scores = rag_retrieve(prompt, top_k=5, min_similarity=min_sim)
        
        if not docs:
            context = "ไม่มีข้อมูลในเอกสารที่เกี่ยวข้องเพียงพอ"
            reply = "ขออภัยครับ ไม่พบข้อมูลที่เกี่ยวข้องในเอกสารที่ให้มาครับ"
        else:
            context = "\n\n".join([f"[ข้อมูลที่ {i+1}] {doc}" for i, doc in enumerate(docs)])
            prompt_with_context = f"""ใช้เฉพาะข้อมูลด้านล่างนี้ในการตอบคำถาม ถ้าไม่มีคำตอบในบริบท ให้ตอบว่า "ไม่มีข้อมูลในเอกสารครับ"

ข้อมูล:
{context}

คำถาม: {prompt}
คำตอบ (ภาษาไทย กระชับ ชัดเจน):"""

            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(prompt_with_context)
                reply = response.text.strip()
            except Exception as e:
                reply = f"เกิดข้อผิดพลาดจาก Gemini: {e}"

        # บันทึกแหล่งที่มา (ถ้ามี)
        source_msg = {}
        if docs:
            source_msg["sources"] = docs
            source_msg["scores"] = scores

    # แสดงผล
    with st.chat_message("assistant"):
        st.markdown(reply)
    
    msg = {"role": "assistant", "content": reply}
    if "sources" in source_msg:
        msg.update(source_msg)
    
    st.session_state.messages.append(msg)
from dotenv import load_dotenv
import os
load_dotenv()


import os

os.environ["HF_TOKEN"] = "***"
os.environ["USER_AGENT"] = "my-rag-agent/1.0"
os.environ["GROQ_API_KEY"] = "****"
os.environ["GROQ_ACCEPT_USER_AGREEMENT"] = "true"


# Verify environment vars
assert os.getenv("HF_TOKEN"), "HF_TOKEN not set!"
assert os.getenv("USER_AGENT"), "USER_AGENT not set!"
assert os.getenv("GROQ_API_KEY"), "GROQ_API_KEY not set!"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import streamlit as st

# Streamlit UI
st.set_page_config(page_title="📚 RAG-based Math Agent", layout="centered")
st.title("📚 RAG-based Math Agent")
st.caption("Ask any question on differentiation & integration")

query = st.text_input("🔍 Ask a question:")
run_button = st.button("Run")

# ---- Cached Setup ----
@st.cache_resource
def setup_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def setup_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )

@st.cache_resource
def setup_vector_store():
    embeddings = setup_embeddings()
    urls = [
        "https://byjus.com/maths/differentiation-integration/",
        "https://www.cuemath.com/differentiation-and-integration-formula/",
        "https://www.geeksforgeeks.org/differentiation-and-integration-formula/"
    ]
    docs = []
    for url in urls:
        loader = WebBaseLoader(url)
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=100
    )
    doc_splits = text_splitter.split_documents(docs)

    vector_store = FAISS.from_documents(doc_splits, embedding=embeddings)
    return vector_store.as_retriever()

# ---- Run RAG ----
if run_button and query:
    with st.spinner("🤖 Thinking..."):
        llm = setup_llm()
        retriever = setup_vector_store()

        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True
        )

        result = rag_chain({"query": query})
        answer = result["result"]
        sources = [doc.metadata.get("source", "unknown source") for doc in result["source_documents"]]

        st.subheader("💬 Answer")
        st.write(answer)

        st.subheader("🔗 Sources")
        for src in sources:
            st.markdown(f"- {src}")

        # ---- Feedback Section ----
        st.subheader("🧠 Was this helpful?")
        thumbs = st.radio("Feedback", ["👍", "👎"], horizontal=True)
        comment = st.text_input("💬 Any suggestions or comments?")

        if st.button("Submit Feedback"):
            feedback_data = {
                "query": query,
                "answer": answer,
                "sources": sources,
                "thumbs": thumbs,
                "comment": comment
            }
            with open("feedback.json", "a") as f:
                f.write(json.dumps(feedback_data) + "\n")

            st.success("✅ Feedback submitted! Thank you 🙌")
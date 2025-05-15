
# 🤖 Agentic-RAG Math Agent with Human-in-the-Loop

A powerful Retrieval-Augmented Generation (RAG) based system tailored for answering math queries using a hybrid approach—combining knowledge base retrieval, web search fallback, symbolic computation, and human-in-the-loop (HIIT) review.

---

## 📌 Features

- 🔒 **Input & Output Guardrails** for prompt moderation and safety
- 📚 **RAG Architecture** using FAISS vector store and HuggingFace Embeddings
- 🌐 **Web Search Fallback** via DuckDuckGo or Tavily APIs
- 🤝 **Human-in-the-Loop** review for low-confidence or ambiguous answers
- 🧮 **Symbolic Math Solving** powered by `SymPy`
- 📊 [Bonus] Benchmarking capability using **JEE Math Bench**

---

## 🧠 Architecture Diagram

![Agentic-RAG System](A_flowchart_illustrates_an_Agentic-RAG_System_arch.png)

---

## 🛠️ Installation

```bash
git clone https://github.com/your-username/agentic-math-rag.git
cd agentic-math-rag
pip install -r requirements.txt




##...............


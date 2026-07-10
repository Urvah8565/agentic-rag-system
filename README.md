# Agentic RAG System

An AI-powered Streamlit application that uses Retrieval-Augmented 
Generation (RAG) and a LangGraph ReAct agent to extract insights 
from business documents and perform mathematical trend analysis.

---

## Features

- Upload your own `.txt` or `.pdf` file
- Extract relevant data using RAG (FAISS + HuggingFace embeddings)
- LangGraph ReAct agent dynamically routes between document 
  retrieval and numerical analysis tools
- Built-in linear regression tool for sales trend detection 
  and next-month forecasting
- Out-of-scope query rejection — agent declines questions not 
  answerable from the document
- Real-time analysis with Gemini API (BYOK – Bring Your Own Key)
- Graceful error handling for API quota limits, invalid keys, 
  and network failures

---

## Live App :
https://agentic-rag-system-kscuctfsngsrc7evpqk5bj.streamlit.app/

## How It Works

1. User uploads a `.txt` or `.pdf` business document 
   (or uses default data)
2. RAG pipeline chunks and embeds the document into FAISS 
   vector store
3. LangGraph ReAct agent decides:
   - When to use the document retrieval tool (factual questions)
   - When to use the linear regression tool (trend/forecast questions)
   - When to reject the query (out-of-scope questions)
4. Final output is generated with full reasoning trace

---

## Model Performance

Validated across 10–12 business documents and multiple query types:
- Targeted retrieval queries
- Full business report generation (summary, trend, forecast, 
  recommendations)
- Out-of-scope query rejection

---

## Tech Stack

- Python
- Streamlit
- LangChain
- LangGraph
- FAISS (Vector Database)
- HuggingFace Embeddings (all-MiniLM-L6-v2)
- Google Gemini API (gemini-2.5-flash)
- pdfplumber (PDF text extraction)

---

## Installation & Setup

```bash
git clone https://github.com/Urvah8565/agentic-rag-system
cd agentic-rag-system
pip install -r requirements.txt
streamlit run app.py
```

## API Key Setup

> **Important Security Note:**
> - Enter your **Google Gemini API Key** in the sidebar.
> - This project uses **BYOK (Bring Your Own Key)**.
> - No API key is stored in the code (secure approach).

## Project Developer
[Urvah Mansuri]

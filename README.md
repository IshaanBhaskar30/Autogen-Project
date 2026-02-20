# 📚 Multi-Agent Literature Review Assistant

A Groq-powered multi-agent research assistant built using Microsoft AutoGen (v0.4+) that autonomously searches and summarizes academic papers from arXiv.

## 🚀 What It Does

This system simulates a collaborative AI workflow:

  - 🔎 Search Agent – Generates optimized queries and retrieves research papers from arXiv

  - 🧠 Summarizer Agent – Produces a structured literature review in Markdown

  - ⚡ Groq (LLaMA 3) – High-speed inference via OpenAI-compatible endpoint

  - 🖥 Streamlit UI – Interactive interface with real-time streaming output

Users provide:

  - Research topic

  - Number of papers

  - Their own Groq API key (BYO secure setup)

## 🏗 Key Concepts Demonstrated

  - Multi-agent orchestration (AutoGen RoundRobinGroupChat)

  - Tool calling with external APIs (arXiv integration)

  - Non-OpenAI model configuration using OpenAI-compatible endpoints

  - Async streaming architecture

  - Controlled output constraints (exact paper count enforcement)

### 🛠 Tech Stack

Python · Microsoft AutoGen (v0.4+) · Groq API · Streamlit · arXiv API

### 🎯 Why This Project Matters

This project demonstrates practical understanding of:

  - Agent-based LLM systems

  - Tool-augmented AI workflows

  - Model provider abstraction (Groq integration)

  - Structured AI reasoning pipelines

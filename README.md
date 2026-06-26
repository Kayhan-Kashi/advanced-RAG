# ⚡ Advanced Event-Driven RAG System with Hybrid Retrieval running on Local LLM Models

**An Advanced Event-Driven RAG System with Hybrid Retrieval (FAISS + MMR + BM25 + BGE Reranker + HyDE) using Event-Driven Architecture (Microservices with Kafka Message Broker) running on Local LLM Models (OLLAMA)**

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)](https://fastapi.tiangolo.com/)
[![FAISS](https://img.shields.io/badge/FAISS-MMR-purple)](https://github.com/facebookresearch/faiss)
[![BM25](https://img.shields.io/badge/BM25-Sparse-yellow)](#)
[![BGE](https://img.shields.io/badge/BGE-Reranker-blue)](https://github.com/FlagOpen/FlagEmbedding)
[![HyDE](https://img.shields.io/badge/HyDE-Query%20Transform-pink)](#)
[![Kafka](https://img.shields.io/badge/Message%20Broker-Kafka-red)](https://kafka.apache.org/)
[![Microservices](https://img.shields.io/badge/Architecture-Microservices-purple)](#)
[![OLLAMA](https://img.shields.io/badge/Local%20Models-OLLAMA-orange)](https://ollama.com/)
[![React](https://img.shields.io/badge/React-18-blue)](https://reactjs.org/)
[![Docker](https://img.shields.io/badge/Docker-24.0-blue)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Hybrid Retrieval Pipeline](#hybrid-retrieval-pipeline)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Environment Variables](#environment-variables)
- [HyDE Configuration](#hyde-configuration)
- [API Endpoints](#api-endpoints)
- [WebSocket Streaming](#websocket-streaming)
- [FastAPI Backend](#fastapi-backend)
- [Privacy First](#privacy-first)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

An event-driven **RAG (Retrieval-Augmented Generation)** system built with:

- 🔄 **Event-Driven Architecture** with asynchronous communication
- 📨 **Kafka Message Broker** for reliable event streaming
- 🏗️ **Microservices** for decoupled, scalable services
- 🎯 **Hybrid Retrieval** using dense + sparse search with diversity
- 🎯 **Precision Reranking** using cross-encoder for final relevance scoring
- 🧠 **HyDE** (Hypothetical Document Embeddings) for query transformation
- 🤖 **Local Models** via OLLAMA - 100% privacy-first, zero cloud costs

---

## 🎯 Key Features

### 1. Event-Driven Microservices
- **Decoupled Services** - Independent, scalable microservices
- **Async Communication** - Non-blocking Kafka events
- **Fault Tolerance** - Service isolation
- **Horizontal Scaling** - Scale services independently

### 2. Hybrid Retrieval Pipeline
- **FAISS + MMR** - Dense semantic retrieval with diversity
- **BM25** - Sparse lexical keyword matching
- **BGE Reranker** - Cross-encoder precision scoring

### 3. HyDE (Hypothetical Document Embeddings)
- **Query Transformation** - Generates hypothetical documents for better retrieval
- **Improved Semantic Matching** - Bridges vocabulary gap between queries and documents
- **Disabled by default** - Enable only when needed

### 4. Local Models (Privacy First)
- **OLLAMA** - Run Gemma3, Llama3, Mistral locally
- **Jina Embeddings v3** - Local dense embeddings
- **BGE Reranker** - Local cross-encoder reranking
- **Zero Cloud** - No external API calls

### 5. Modern UI
- **React + TypeScript** - Modern, responsive interface
- **WebSocket Streaming** - Real-time token-by-token responses
- **Document Management** - Upload, delete, and track document status
- **Conversation History** - Persistent chat history

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         React Frontend                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      FastAPI Gateway                                   │
│                   (REST API + Kafka Producer)                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   Apache Kafka Message Broker                         │
│                    ⚡ Event-Driven Communication                       │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Topics:                                                       │   │
│  │  ┌───────────────────────┐  ┌───────────────────────────────┐ │   │
│  │  │ prompt-requested      │  │ prompt-completed              │ │   │
│  │  └───────────────────────┘  └───────────────────────────────┘ │   │
│  │  ┌───────────────────────┐  ┌───────────────────────────────┐ │   │
│  │  │ document-uploaded     │  │ document-embedding-done       │ │   │
│  │  └───────────────────────┘  └───────────────────────────────┘ │   │
│  │  ┌───────────────────────────────────────────────────────────┐ │   │
│  │  │ prompt-answer-chunk-streamed (for real-time streaming)   │ │   │
│  │  └───────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐
│  LLM Service    │    │  Chat Service   │    │  Ingestion Service  │
│   (Consumer)    │    │   (Consumer)    │    │    (Consumer)       │
│                 │    │                 │    │                     │
│  🧠 OLLAMA      │    │  💬 Chat Logic  │    │  📄 Document        │
│  (Local Models) │    │                 │    │     Processing      │
└─────────────────┘    └─────────────────┘    └─────────────────────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   ⚡ Hybrid Retrieval Pipeline                         │
│                                                                         │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐ │
│  │   FAISS + MMR    │  │   BM25 (Sparse)  │  │  BGE Reranker        │ │
│  │   Dense Search   │ +│   Lexical Search │ →│  Cross-Encoder       │ │
│  │   Semantic +     │  │   Keyword        │  │  Precision Scoring   │ │
│  │   Diversity      │  │   Matching       │  │  Final Ranking       │ │
│  └──────────────────┘  └──────────────────┘  └──────────────────────┘ │
│                                                                         │
│  🔄 HyDE (Hypothetical Document Embeddings) for Query Transformation   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Hybrid Retrieval Pipeline

### Stage 1: HyDE (Query Transformation) - Optional
- Generates a hypothetical document from the user query
- Transforms the query into a document-like format for better retrieval
- Improves semantic matching with documents
- **Disabled by default** - Enable via `USE_HYDE=True`

### Stage 2: FAISS + MMR (Dense Retrieval with Diversity)
- FAISS for fast similarity search on dense embeddings
- MMR (Maximum Marginal Relevance) for diverse, non-redundant results
- Output - Semantically relevant + diverse candidate set

### Stage 3: BM25 (Sparse Lexical Retrieval)
- BM25 for keyword-based lexical matching
- Output - Lexically relevant candidates

### Stage 4: BGE Reranker (Cross-Encoder Precision)
- BGE Reranker v2 for final precision scoring
- Cross-encoder for deep relevance assessment
- Output - Highly relevant, precision-ranked final results

---

## 🔧 Technology Stack

| Component | Technology | Role |
|-----------|------------|------|
| **Message Broker** | Apache Kafka | Async event communication |
| **LLM** | OLLAMA (Local) | Text generation + HyDE |
| **Query Transform** | HyDE | Hypothetical Document Embeddings |
| **Dense Search** | FAISS + MMR | Semantic retrieval with diversity |
| **Sparse Search** | BM25 | Lexical keyword matching |
| **Reranker** | BGE Reranker v2 | Cross-encoder precision |
| **Embeddings** | Jina AI v3 | Dense vector encoding |
| **Backend** | Python 3.11 + FastAPI | Kafka producers/consumers |
| **Frontend** | React + TypeScript | Modern UI |
| **WebSocket** | FastAPI WebSockets | Real-time streaming |
| **Database** | SQLite | Conversation storage |
| **Orchestration** | Docker Compose | Container management |

---

## 📁 Project Structure

```
advanced-RAG/
├── backend/
│   ├── services/
│   │   ├── chat-service/
│   │   │   ├── Dockerfile
│   │   │   ├── requirements.txt
│   │   │   └── src/
│   │   │       ├── api/
│   │   │       ├── handlers/
│   │   │       ├── services/
│   │   │       ├── workers/
│   │   │       └── main.py
│   │   └── llm-service/
│   │       ├── Dockerfile
│   │       ├── requirements.txt
│   │       └── src/
│   │           ├── services/
│   │           ├── core/
│   │           ├── handlers/
│   │           ├── workers/
│   │           └── main.py
│   └── shared/
│       ├── common/
│       │   ├── events/
│       │   └── kafka/
│       └── utils/
├── frontend/
│   └── rag-react-app/
│       ├── src/
│       ├── public/
│       └── package.json
├── data/
│   ├── uploads/
│   ├── faiss_index/
│   └── bm25_index/
├── models/
│   ├── .cache/
│   ├── BAAI/
│   │   └── models--BAAI--bge-reranker-v2-m3/
│   ├── bm25_index/
│   ├── dynamic_modules/
│   ├── faiss_index/
│   ├── hf_cache/
│   └── snapshot/
│       └── jina-embeddings-v3/
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- [OLLAMA](https://ollama.com/) installed locally
- Make sure ports 8001, 3000, 9092, 9000 are available

### 1. Install OLLAMA

Download and install OLLAMA from [https://ollama.com/](https://ollama.com/)

### 2. Download a Model

```bash
ollama pull gemma3:12b
# or
ollama pull llama3:8b
# or
ollama pull mistral
```

### 3. Start OLLAMA Service

```bash
ollama serve
```

### 4. Clone the Repository

```bash
git clone https://github.com/Kayhan-Kashi/advanced-RAG.git
cd advanced-RAG
```

### 5. Start All Services

```bash
docker-compose up -d
```

### 6. Access the Application

- **Frontend**: http://localhost:3000
- **Chat Service API**: http://localhost:8001
- **Kafka UI**: http://localhost:9000

---

## 📊 Environment Variables

```yaml
rag-llm-service:
  environment:
    # Kafka
    - KAFKA_BOOTSTRAP_SERVERS=kafka:29092
    - CONSUMER_GROUP=llm-worker-group
    
    # OLLAMA
    - OLLAMA_MODEL=gemma3:12b
    - OLLAMA_BASE_URL=http://host.docker.internal:11434
    - OLLAMA_TEMPERATURE=0.3
    
    # Embedding Model
    - MODEL_PATH=/app/models/snapshot/jina-embeddings-v3
    
    # HuggingFace
    - HF_HUB_OFFLINE=0
    - HF_HUB_ENABLE_OFFLINE=0
    - TRANSFORMERS_OFFLINE=0
    - HF_HUB_DISABLE_SYMLINKS_WARNING=1
    - HF_HOME=/app/models/hf_cache
    
    # HyDE
    - USE_HYDE=False
```

---

## 🧠 HyDE Configuration

### What is HyDE?

HyDE (Hypothetical Document Embeddings) is a query transformation technique that:
1. Takes the user's question
2. Asks the LLM to generate a hypothetical document that would contain the answer
3. Uses this hypothetical document for semantic search instead of the original query
4. Improves retrieval quality by bridging the vocabulary gap between queries and documents

### Enabling HyDE

```yaml
environment:
  - USE_HYDE=True  # Enable HyDE
```

### When to Enable HyDE

| Scenario | Recommendation |
|----------|----------------|
| Short queries (1-3 words) | Enable HyDE |
| Ambiguous queries | Enable HyDE |
| Technical/domain-specific questions | Enable HyDE |
| Long, well-formed questions | Disable HyDE |
| Low latency requirements | Disable HyDE |
| Simple, direct questions | Disable HyDE |

---

## 📊 Event Flow

### User Request Flow
```
User → Frontend → Gateway → Kafka → LLM Service → Hybrid Retrieval → 
WebSocket → Frontend (Real-time streaming)
```

### Document Ingestion Flow
```
User → Frontend → Gateway → Kafka → LLM Service → 
Process Document (Chunk + Embed + Index) → 
Kafka → Chat Service → Database → Frontend
```

---

## 🌐 API Endpoints

### Conversations

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/conversation/new` | Create a new conversation |
| GET | `/conversation/user/{user_id}` | Get user conversations |
| GET | `/conversation/{conversation_id}` | Get conversation details |
| DELETE | `/conversation/{conversation_id}` | Delete a conversation |

### Documents

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/documents/upload` | Upload a document |
| GET | `/documents/` | Get all documents |
| GET | `/documents/{document_id}/status` | Get document status |
| DELETE | `/documents/{document_id}` | Delete a document |

### WebSocket

| Endpoint | Description |
|----------|-------------|
| `/ws/{user_id}` | WebSocket connection for real-time chat |

---

## 🔌 WebSocket Streaming

### Sending a Chat Message
```json
{
    "type": "chat",
    "conversation_id": "abc-123",
    "prompt": "What is RAG?",
    "file_ids": ["doc-1", "doc-2"]
}
```

### Receiving a Chunk
```json
{
    "type": "answer_chunk",
    "chunk": "RAG stands for ",
    "chunk_index": 0,
    "is_last": false
}
```

---

## 🚀 FastAPI Backend

The backend is built with **FastAPI** and provides:

- Automatic API Documentation at `/docs` (Swagger UI) and `/redoc`
- WebSocket Support for real-time streaming
- Pydantic Models for request/response validation
- Dependency Injection with `fastapi-injector`

### Health Check

```bash
curl http://localhost:8001/health
# {"status": "online", "consumer_running": true}
```

---

## 🔒 Privacy First

- ✅ 100% Local - No external API calls
- ✅ Zero Cloud Costs - No per-token or per-request fees
- ✅ Air-Gap Ready - Works in isolated environments
- ✅ Data Sovereignty - Complete control over your data
- ✅ No Data Leakage - Your documents never leave your infrastructure

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- [OLLAMA](https://ollama.com/) for local LLM inference
- [Jina AI](https://jina.ai/) for embeddings
- [BAAI](https://www.baai.ac.cn/) for BGE reranker
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [Apache Kafka](https://kafka.apache.org/) for event streaming
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework

---

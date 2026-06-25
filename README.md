Here's the updated README with the corrected title and models folder structure:

```markdown
# ⚡ Advanced Event-Driven RAG System

**Event-Driven RAG System Hybrid Retrieval (FAISS + MMR + BM25 + BGE Reranker + HyDE) using Event-Driven Architecture (Microservices with Kafka Message Broker) running on Local Models (OLLAMA)**

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
- [API Endpoints](#api-endpoints)
- [WebSocket Streaming](#websocket-streaming)
- [FastAPI Backend](#fastapi-backend)
- [Privacy First](#privacy-first)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

A production-grade, event-driven **RAG (Retrieval-Augmented Generation)** system built with:

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

### Stage 1: HyDE (Query Transformation)
- **HyDE** generates a hypothetical document from the user query
- Transforms the query into a document-like format for better retrieval
- Improves semantic matching with documents

### Stage 2: FAISS + MMR (Dense Retrieval with Diversity)
- **FAISS** for fast similarity search on dense embeddings
- **MMR** (Maximum Marginal Relevance) for diverse, non-redundant results
- **Output** - Semantically relevant + diverse candidate set

### Stage 3: BM25 (Sparse Lexical Retrieval)
- **BM25** for keyword-based lexical matching
- **Output** - Lexically relevant candidates

### Stage 4: BGE Reranker (Cross-Encoder Precision)
- **BGE Reranker v2** for final precision scoring
- Cross-encoder for deep relevance assessment
- **Output** - Highly relevant, precision-ranked final results

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
advanced-rag/
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
│   ├── .cache/                  # HuggingFace cache
│   ├── BAAI/                    # BGE reranker model
│   │   └── models--BAAI--bge-reranker-v2-m3/
│   ├── bm25_index/              # BM25 index files
│   ├── dynamic_modules/         # Dynamic modules for models
│   ├── faiss_index/             # FAISS vector index
│   ├── hf_cache/                # HuggingFace cache
│   └── snapshot/                # Jina embeddings model
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
- Make sure ports 8000, 3000, 9092, 9000 are available

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/advanced-rag.git
cd advanced-rag
```

### 2. Environment Configuration

```bash
cp .env.example .env
# Edit .env with your configuration
```

### 3. Start All Services

```bash
docker-compose up -d
```

### 4. Pull Models

```bash
# Pull OLLAMA models
docker-compose exec llm-service ollama pull gemma3:12b

# Or pull Llama3
docker-compose exec llm-service ollama pull llama3:8b
```

### 5. Access the Application

- **Frontend**: http://localhost:3000
- **Chat Service API**: http://localhost:8000/docs
- **Kafka UI**: http://localhost:9000

---

## 📊 Event Flow

### 1. User Request Flow
```
User → Frontend → Gateway (Producer) 
                → Kafka Topic: "prompt-requested"
                → LLM Service (Consumer)
                → Hybrid Retrieval Pipeline
                → Kafka Topic: "prompt-answer-chunk-streamed"
                → WebSocket → Frontend (Real-time streaming)
                → Kafka Topic: "prompt-completed"
                → Chat Service (Consumer)
                → Database
```

### 2. Document Ingestion Flow
```
User → Frontend → Gateway (Producer)
                → Kafka Topic: "document-uploaded"
                → LLM Service (Consumer)
                → Process Document (Chunk + Embed + Index)
                → Kafka Topic: "document-embedding-done"
                → Chat Service (Consumer)
                → Database → Frontend
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

The system supports real-time token-by-token streaming via WebSockets.

### WebSocket Message Types

#### Sending a Chat Message
```json
{
    "type": "chat",
    "conversation_id": "abc-123",
    "prompt": "What is RAG?",
    "file_ids": ["doc-1", "doc-2"]
}
```

#### Register Conversation
```json
{
    "type": "register_conversation",
    "conversation_id": "abc-123"
}
```

#### Receiving a Chunk
```json
{
    "type": "answer_chunk",
    "chunk": "RAG stands for ",
    "chunk_index": 0,
    "is_last": false
}
```

#### Receiving Final Chunk
```json
{
    "type": "answer_chunk",
    "chunk": "Retrieval Augmented Generation",
    "chunk_index": 5,
    "is_last": true
}
```

---

## 🚀 FastAPI Backend

The backend is built with **FastAPI** and provides:

- 📝 **Automatic API Documentation** at `/docs` (Swagger UI) and `/redoc`
- 🔌 **WebSocket Support** for real-time streaming
- 🧩 **Pydantic Models** for request/response validation
- 🏗️ **Dependency Injection** with `fastapi-injector`

### Interactive API Docs

Once running, visit:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Example API Calls

#### Create a Conversation
```bash
curl -X POST "http://localhost:8000/conversation/new" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "12345678-1234-5678-1234-567812345678"}'
```

#### Upload a Document
```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -F "file=@document.pdf" \
  -F "user_id=12345678-1234-5678-1234-567812345678"
```

#### Get Document Status
```bash
curl "http://localhost:8000/documents/{document_id}/status"
```

### WebSocket Connection

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/12345678-1234-5678-1234-567812345678');

ws.send(JSON.stringify({
    type: 'chat',
    conversation_id: 'conversation-id',
    prompt: 'What is RAG?',
    file_ids: ['doc-1', 'doc-2']
}));
```

### Health Check

```bash
curl http://localhost:8000/health
# {"status": "online", "consumer_running": true}
```

---

## 🔒 Privacy First

- ✅ **100% Local** - No external API calls
- ✅ **Zero Cloud Costs** - No per-token or per-request fees
- ✅ **Air-Gap Ready** - Works in isolated environments
- ✅ **Data Sovereignty** - Complete control over your data
- ✅ **No Data Leakage** - Your documents never leave your infrastructure

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⭐ Support

If you find this project useful, please give it a star! ⭐

---

## 🙏 Acknowledgments

- [OLLAMA](https://ollama.com/) for local LLM inference
- [Jina AI](https://jina.ai/) for embeddings
- [BAAI](https://www.baai.ac.cn/) for BGE reranker
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [Apache Kafka](https://kafka.apache.org/) for event streaming
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework

---

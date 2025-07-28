# ChatBot
## Overview
This project is a robust backend application built with FastAPI that enables a smart, conversational virtual assistant capable of understanding and responding to user queries with contextual awareness. 
The assistant uses local large language models (LLMs) through Ollama for intent detection and language generation. Based on the identified intent, it retrieves relevant information from different knowledge sources—such as product details, FAQs, and navigation guides—using a hybrid retrieval approach that combines keyword-based search (BM25) with semantic vector search (ChromaDB). 
Retrieved results are further refined through a CrossEncoder-based reranking mechanism to ensure accuracy and relevance. The system maintains conversational context using Redis, allowing it to handle follow-up questions and session-based interactions effectively.

## Features
Intent Detection using Local LLM
Classifies user queries into intents such as product recommendation, FAQ, returns, or navigation help using a local language model via Ollama.

Hybrid Document Retrieval
Combines BM25 (keyword-based) and ChromaDB vector-based retrieval for improved coverage and semantic relevance.

Reranking with CrossEncoder
Uses a BGE-based CrossEncoder to rerank retrieved results, ensuring the most relevant content is prioritized.

Strictly Grounded Responses
Responses are generated only from retrieved content—no hallucination or fabricated information.

Session Memory with Redis
Maintains conversational history and supports follow-up queries by storing session data in Redis with TTL-based cleanup.

Multi-Domain Knowledge Support
Routes queries to different retrievers for FAQs, product information, and website navigation based on detected intent.

FastAPI-Powered Backend
Asynchronous, high-performance REST API server with clean endpoint structure for easy integration.

Follow-Up Query Handling
Tracks previous intents and queries to respond to follow-ups naturally within the context of the conversation.

## Tech Stack, Libraries & Frameworks
### Language Models & Embeddings
Ollama – Serves local LLMs (e.g., LLaMA2, Mistral) for intent detection and response generation

OllamaEmbeddings – Generates vector embeddings for document chunks using local embedding models

### Retrieval & RAG (Retrieval-Augmented Generation)
LangChain – Core framework for building prompt chains, handling memory, retrievers, and output parsing

ChromaDB – Vector database used for semantic document retrieval

BM25Retriever – Classic keyword-based search to complement vector retrieval

EnsembleRetriever – Combines BM25 and Vector retrieval with custom weighting

CrossEncoderReranker (BGE-Reranker) – Reranks retrieved documents for relevance using a Hugging Face CrossEncoder model

ContextualCompressionRetriever – Wraps retrievers with rerankers to return only top relevant chunks

### Backend & API
FastAPI – High-performance, asynchronous Python web framework for building the backend API

Uvicorn – ASGI server for running the FastAPI app

Pydantic – Data validation and parsing for API request/response models

CORS Middleware – Enables safe cross-origin frontend requests

### Memory & Session Handling
Redis – In-memory data store used to persist chat history and track session-specific data like last intent or query

RedisChatMessageHistory (LangChain) – Manages message history in Redis for long-running chat interactions

RunnableWithMessageHistory – Integrates memory into the LangChain pipeline for contextual responses

### Data Loading & Processing
TextLoader (LangChain) – Loads plain text documents into the pipeline

RecursiveCharacterTextSplitter – Splits documents into overlapping chunks optimized for retrieval

## Deployment Guide
Follow these steps to set up and run the backend application:

### 1. Install Ollama
Install Ollama to run local LLMs and embedding models
### 2. Pull Required Models
Pull the required language model (for intent detection & generation) and embedding model (for vector search):
For LLM (e.g., LLaMA2 or Mistral)
```bash
ollama pull llama2
```

For Embeddings (e.g., Mistral, Nomic, etc.)
```bash
ollama pull mistral
```

Update the model names in your code:
```python
llm = OllamaLLM(model="llama2")
embedding_model = OllamaEmbeddings(model="mistral")
```

### 3. Install Python Dependencies
Create a virtual environment (optional but recommended):
```bash
# For Windows
python -m venv venv
venv\Scripts\activate
```

Install required libraries:
```bash
pip install -r requirements.txt
```

### 4. Set Up Redis (Cloud Recommended)
You can use a cloud-based Redis service
Update this line in your code with the Redis connection URL:
```python
redis_url = "redis://<your-redis-url>"
```

### 5. Run the FastAPI App
Start the backend server with:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```


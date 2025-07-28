import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_loaders import TextLoader

# Loading text files
loader = TextLoader("data/QA_format_FAQ.txt", encoding="utf-8")
documents = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    add_start_index=True,
)
docs = splitter.split_documents(documents)

# embeddings model
embedding_model = OllamaEmbeddings(model="<embedding-model>")

# vector store (no persist)
vector_store = Chroma.from_documents(docs, embedding_model)

# BM25 retriever
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 6

# Vector retriever
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 6})

# Hybrid retriever
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7],
)

# Reranker (CrossEncoder)-->BGE reranker base
reranker_model = HuggingFaceCrossEncoder(model_name="models/bge-reranker-base")
compressor = CrossEncoderReranker(model=reranker_model, top_n=3)

# Final retriever with compression
compression_retriever_books = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=hybrid_retriever
)


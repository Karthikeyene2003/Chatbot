import json
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Load JSON data
with open("D:data/navigating_websites.json", "r") as f:
    data = json.load(f)

# Convert to LangChain Documents
docs = [
    Document(
        page_content=item["purpose"],
        metadata={
            "name": item["name"],
            "url": item["url"],
            "class": item["class"],
        }
    )
    for item in data
]

# Initialize embeddings
embedding_model = OllamaEmbeddings(model="<emdedding-model>")

# Load or create vectorstore

vector_url_store = Chroma.from_documents(docs, embedding_model)

# BM25 keyword retriever
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 5

# Semantic (vector) retriever
vector_retriever_url= vector_url_store.as_retriever(search_kwargs={"k": 5})

# Hybrid retriever using EnsembleRetriever
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever_url],
    weights=[0.7, 0.3] 
)
#reranker
reranker_model_path = "models/bge-reranker-base"
model = HuggingFaceCrossEncoder(model_name=reranker_model_path)
compressor = CrossEncoderReranker(model=model, top_n=2)
compression_retriever_url = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=hybrid_retriever
)


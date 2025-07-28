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
with open("data/products_services.json", "r") as f:
    data = json.load(f)

# Convert to LangChain Documents
docs = [
    Document(
        page_content=item["purpose"],
        metadata={
            "name": item["name"],
            "url": item["url"],
            "class": item["class"],
            "type": item.get("type")
        }
    )
    for item in data
]

# Initialize embeddings
embedding_model = OllamaEmbeddings(model="<embedding-model>")

# Load or create vectorstore
vector_product_store = Chroma.from_documents(docs, embedding_model)

# BM25 keyword retriever
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 4

# Semantic (vector) retriever
vector_retriever_sale= vector_product_store.as_retriever(search_kwargs={"k": 4})

# Hybrid retriever using EnsembleRetriever
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever_sale],
    weights=[0.5, 0.5]  
)
#reranker
model = HuggingFaceCrossEncoder(model_name="models/bge-reranker-base")
compressor = CrossEncoderReranker(model=model, top_n=3)
compression_retriever_sales = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=hybrid_retriever
)


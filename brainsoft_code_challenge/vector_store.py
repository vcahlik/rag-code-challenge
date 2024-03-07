import chromadb
from langchain_openai import OpenAIEmbeddings


class VectorStore:
    def __init__(self):
        self._embedder = None
        self._chromadb_collection = None

    def get_embedder(self):
        if self._embedder is None:
            self._embedder = OpenAIEmbeddings(model="text-embedding-3-large")
        return self._embedder

    def get_chromadb_collection(self):
        if self._chromadb_collection is None:
            chroma_client = chromadb.PersistentClient(path="../chromadb")
            self._chromadb_collection = chroma_client.get_collection(name="documentation")
        return self._chromadb_collection

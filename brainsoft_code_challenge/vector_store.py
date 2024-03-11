from collections.abc import Mapping

import chromadb
from langchain_openai import OpenAIEmbeddings

MetadataType = Mapping[str, str | int | float | bool]


class VectorStore:
    """
    A class to manage the embeddings and the vector database.
    """

    def __init__(self) -> None:
        self._embedder: OpenAIEmbeddings | None = None
        self._chromadb_collection: chromadb.Collection | None = None

    def get_embedder(self) -> OpenAIEmbeddings:
        if self._embedder is None:
            self._embedder = OpenAIEmbeddings(model="text-embedding-3-large")
        return self._embedder

    def get_chromadb_collection(self) -> chromadb.Collection:
        if self._chromadb_collection is None:
            chroma_client = chromadb.PersistentClient(path="../chromadb")
            self._chromadb_collection = chroma_client.get_collection(name="documentation")
        return self._chromadb_collection

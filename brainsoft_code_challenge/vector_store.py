import chromadb
from langchain_openai import OpenAIEmbeddings


def get_embedder():
    return OpenAIEmbeddings(model="text-embedding-3-large")


def get_chromadb_collection():
    chroma_client = chromadb.PersistentClient(path="../chromadb")
    return chroma_client.get_collection(name="documentation")

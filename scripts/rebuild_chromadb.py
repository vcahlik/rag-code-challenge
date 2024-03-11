import argparse
import json
from collections.abc import Sequence
from typing import cast
from uuid import uuid4

import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm.autonotebook import tqdm

from brainsoft_code_challenge.tokenizer import count_tokens
from brainsoft_code_challenge.vector_store import MetadataType, VectorStore

vector_store = VectorStore()


def upsert_to_index(texts: Sequence[str], metadatas: list[MetadataType], collection: chromadb.Collection) -> None:
    texts = list(texts)
    ids = [str(uuid4()) for _ in range(len(texts))]
    embeddings = cast(list[Sequence[float]], vector_store.get_embedder().embed_documents(texts))
    collection.add(documents=texts, metadatas=metadatas, ids=ids, embeddings=embeddings)


def rebuild_chromadb(data: Sequence[MetadataType]) -> None:
    chroma_client = chromadb.PersistentClient(path="../chromadb")
    collections = chroma_client.list_collections()
    if "documentation" in [collection.name for collection in collections]:
        chroma_client.delete_collection("documentation")
    collection = chroma_client.create_collection(name="documentation", metadata={"hnsw:space": "ip"})

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=75, length_function=count_tokens, separators=["\n\n", "\n", " ", ""])

    batch_limit = 100
    text_chunks = []
    metadatas = []
    for document in tqdm(data):
        document_text_chunks = text_splitter.split_text(document["content"])
        document_metadatas = []
        for i, text_chunk in enumerate(document_text_chunks):
            embedding_text = f"{document['documentation_url']}: {text_chunk}"
            metadata: MetadataType = {
                "chunk": i,
                "embedding_text": embedding_text,
                **{k: v for k, v in document.items() if v is not None},
            }
            document_metadatas.append(metadata)
        text_chunks.extend(document_text_chunks)
        metadatas.extend(document_metadatas)
        if len(text_chunks) >= batch_limit:
            upsert_to_index(text_chunks, metadatas, collection)
            text_chunks = []
            metadatas = []
    if text_chunks:
        upsert_to_index(text_chunks, metadatas, collection)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, default="split_docs.json", help="Path to input data")
    args = parser.parse_args()

    with open(args.input_path) as f:
        data = json.load(f)
    rebuild_chromadb(data)

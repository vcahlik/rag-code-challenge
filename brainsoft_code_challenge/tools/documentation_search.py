from collections.abc import Sequence
from typing import cast

from langchain.agents import tool
from pydantic.v1 import BaseModel, Field

from brainsoft_code_challenge.config import N_CHROMADB_RESULTS, N_CHROMADB_UNIQUE_RESULTS
from brainsoft_code_challenge.vector_store import MetadataType, VectorStore

vector_store = VectorStore()


def __get_unique_results(results: Sequence[MetadataType], n_results: int) -> list[MetadataType]:
    """
    Documentation documents are split into smaller "splits", which are split into "chunks". Embeddings are calculated chunk-wise,
    so we need to ensure that we only return unique results, where the same document split is not repeated.

    :param results: Results from the vector databaser.
    :param n_results: The number of unique results to return.
    :return: The unique results.
    """
    unique_results: list[MetadataType] = []
    for result in results:
        source_url = result["source_url"]
        split_part = result.get("split_part")

        is_new = True
        for unique_result in unique_results:
            if unique_result["source_url"] == source_url and unique_result.get("split_part") == split_part:
                is_new = False
                break

        if is_new:
            unique_results.append(result)
            if len(unique_results) >= n_results:
                break
    return unique_results


class DocumentationQuery(BaseModel):
    query: str = Field(description="The query to execute")


@tool(args_schema=DocumentationQuery)
def search_documentation(query: str) -> str:
    """Searches the documentation (development version) using a natural language query."""  # Tool description for agent
    query_embeddings = cast(list[Sequence[float]], vector_store.get_embedder().embed_documents([query]))
    metadatas = vector_store.get_chromadb_collection().query(query_embeddings=query_embeddings, n_results=N_CHROMADB_RESULTS, include=["metadatas"])[
        "metadatas"
    ]  # noqa: E501
    if not metadatas:
        return "No results found."
    results = metadatas[0]
    results = __get_unique_results(results, n_results=N_CHROMADB_UNIQUE_RESULTS)
    outputs = []
    for result in results:
        output = f"Documentation page URL: {result['documentation_url']}\n"
        output += str(result["content"])
        outputs.append(output)
    return "\n\n========================================\n\n".join(outputs)

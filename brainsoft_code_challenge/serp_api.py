from langchain_community.utilities import GoogleSerperAPIWrapper


search = GoogleSerperAPIWrapper()


def serp_api_search(query: str, num_results: int):
    results = search.results(query)["organic"][:num_results]
    return [r["link"] for r in results]

import logging
import os

import requests

from brainsoft_code_challenge.utils import is_pytest_running

REQUEST_TIMEOUT_SECONDS = 60


def __get_headers_for_github(github_api_token: str | None) -> dict[str, str]:
    """
    Get the request headers for GitHub API to avoid rate limits.

    :param github_api_token: The GitHub API token to avoid rate limits.
    :return: The request headers for GitHub API.
    """
    if not github_api_token:
        return {}
    return {"Authorization": f"Bearer {github_api_token}"}


def __scrape_documentation(github_api_token: str | None = None) -> list[dict[str, str]]:
    """
    Scrape the documentation from the IBM Generative AI repository.

    :param github_api_token: The GitHub API token to avoid rate limits.
    :return: The scraped documentation.
    """
    url = "https://api.github.com/repos/IBM/ibm-generative-ai/contents/documentation/source"
    response = requests.get(url, headers=__get_headers_for_github(github_api_token), timeout=REQUEST_TIMEOUT_SECONDS)
    data = response.json()
    results = []
    for file in data:
        if file["name"].endswith(".rst"):
            if file["name"] == "404.rst":
                continue
            documentation_url = f"https://ibm.github.io/ibm-generative-ai/main/{file['name'].removesuffix('.rst')}.html"
            if is_pytest_running():
                assert requests.get(documentation_url, timeout=REQUEST_TIMEOUT_SECONDS).status_code == 200  # noqa: PLR2004, S101
            logging.info(f"Found page {documentation_url}")
            source_url = file["download_url"]
            source_response = requests.get(source_url, headers=__get_headers_for_github(github_api_token), timeout=REQUEST_TIMEOUT_SECONDS)
            content = source_response.text
            result = {
                "source_path": file["path"],
                "source_url": source_url,
                "documentation_url": documentation_url,
                "content": content,
                "type": "documentation",
            }
            results.append(result)
    return results


def __scrape_examples_dir(url: str, github_api_token: str | None) -> list[dict[str, str]]:
    """
    Recursively scrape the examples from a directory in the IBM Generative AI repository.
    :param url: The URL of the directory in the repository.
    :param github_api_token: The GitHub API token to avoid rate limits.
    :return: The scraped examples.
    """
    response = requests.get(url, headers=__get_headers_for_github(github_api_token), timeout=REQUEST_TIMEOUT_SECONDS)
    data = response.json()
    results = []
    for file in data:
        if file["type"] == "dir":
            dir_url = file["url"]
            results.extend(__scrape_examples_dir(dir_url, github_api_token))
        elif file["type"] == "file" and file["name"].endswith(".py"):
            if file["name"] == "__init__.py":
                continue
            documentation_url = f"https://ibm.github.io/ibm-generative-ai/main/rst_source/{file['path'].removesuffix('.py').replace('/', '.')}.html"
            if is_pytest_running():
                assert requests.get(documentation_url, timeout=REQUEST_TIMEOUT_SECONDS).status_code == 200  # noqa: PLR2004, S101
            logging.info(f"Found page {documentation_url}")
            source_url = file["download_url"]
            source_response = requests.get(source_url, headers=__get_headers_for_github(github_api_token), timeout=REQUEST_TIMEOUT_SECONDS)
            content = source_response.text
            result = {"source_path": file["path"], "source_url": source_url, "documentation_url": documentation_url, "content": content, "type": "example"}
            results.append(result)
    return results


def __scrape_examples(github_api_token: str | None = None) -> list[dict[str, str]]:
    """
    Scrape the examples from the IBM Generative AI repository.

    :param github_api_token: The GitHub API token to avoid rate limits.
    :return: The scraped examples.
    """
    url = "https://api.github.com/repos/IBM/ibm-generative-ai/contents/examples"
    return __scrape_examples_dir(url, github_api_token)


def scrape_all(github_api_token: str | None = None) -> list[dict[str, str]]:
    """
    Scrape all the documentation and examples from the IBM Generative AI repository.

    :param github_api_token: The GitHub API token to avoid rate limits.
    :return: The scraped documentation and examples.
    """
    if github_api_token is None:
        github_api_token = os.getenv("GITHUB_API_TOKEN")
    results = []
    results.extend(__scrape_documentation(github_api_token))
    results.extend(__scrape_examples(github_api_token))
    return results

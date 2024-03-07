import logging
import os

import requests

from brainsoft_code_challenge.utils import is_pytest_running

REQUEST_TIMEOUT_SECONDS = 60


def get_headers(github_api_token: str):
    if not github_api_token:
        return {}
    return {"Authorization": f"Bearer {github_api_token}"}


def __scrape_documentation(github_api_token: str | None = None):
    url = "https://api.github.com/repos/IBM/ibm-generative-ai/contents/documentation/source"
    response = requests.get(url, headers=get_headers(github_api_token), timeout=REQUEST_TIMEOUT_SECONDS)
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
            source_response = requests.get(source_url, headers=get_headers(github_api_token), timeout=REQUEST_TIMEOUT_SECONDS)
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


def __scrape_examples_dir(url: str, github_api_token: str):
    response = requests.get(url, headers=get_headers(github_api_token), timeout=REQUEST_TIMEOUT_SECONDS)
    data = response.json()
    results = []
    for file in data:
        if file["type"] == "dir":
            dir_url = file["url"]
            results.extend(__scrape_examples_dir(dir_url, github_api_token))
        elif file["type"] == "file":
            if file["name"].endswith(".py"):
                if file["name"] == "__init__.py":
                    continue
                documentation_url = f"https://ibm.github.io/ibm-generative-ai/main/rst_source/{file['path'].removesuffix('.py').replace('/', '.')}.html"
                if is_pytest_running():
                    assert requests.get(documentation_url, timeout=REQUEST_TIMEOUT_SECONDS).status_code == 200  # noqa: PLR2004, S101
                logging.info(f"Found page {documentation_url}")
                source_url = file["download_url"]
                source_response = requests.get(source_url, headers=get_headers(github_api_token), timeout=REQUEST_TIMEOUT_SECONDS)
                content = source_response.text
                result = {"source_path": file["path"], "source_url": source_url, "documentation_url": documentation_url, "content": content, "type": "example"}
                results.append(result)
    return results


def __scrape_examples(github_api_token: str | None = None):
    url = "https://api.github.com/repos/IBM/ibm-generative-ai/contents/examples"
    return __scrape_examples_dir(url, github_api_token)


def scrape_all(github_api_token: str | None = None):
    if github_api_token is None:
        github_api_token = os.getenv("GITHUB_API_TOKEN")
    results = []
    results.extend(__scrape_documentation(github_api_token))
    results.extend(__scrape_examples(github_api_token))
    return results

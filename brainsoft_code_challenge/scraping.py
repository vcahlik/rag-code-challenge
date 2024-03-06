import requests
import logging


def get_headers(github_api_token: str):
    if not github_api_token:
        return {}
    return {"Authorization": f"Bearer {github_api_token}"}


def scrape_documentation(github_api_token: str | None = None):
    url = "https://api.github.com/repos/IBM/ibm-generative-ai/contents/documentation/source"
    response = requests.get(url, headers=get_headers(github_api_token))
    data = response.json()
    results = []
    for file in data:
        if file["name"].endswith(".rst"):
            if file["name"] == "404.rst":
                continue
            documentation_url = f"https://ibm.github.io/ibm-generative-ai/main/{file['name'].removesuffix('.rst')}.html"
            assert requests.get(documentation_url).status_code == 200
            logging.info(f"Found page {documentation_url}")
            source_url = file["download_url"]
            source_response = requests.get(source_url, headers=get_headers(github_api_token))
            content = source_response.text
            result = {
                "source_path": file["path"],
                "source_url": source_url,
                "documentation_url": documentation_url,
                "content": content,
                "type": "documentation"
            }
            results.append(result)
    return results


def __scrape_examples_dir(url: str, github_api_token: str):
    response = requests.get(url, headers=get_headers(github_api_token))
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
                assert requests.get(documentation_url).status_code == 200
                logging.info(f"Found page {documentation_url}")
                source_url = file["download_url"]
                source_response = requests.get(source_url, headers=get_headers(github_api_token))
                content = source_response.text
                result = {
                    "source_path": file["path"],
                    "source_url": source_url,
                    "documentation_url": documentation_url,
                    "content": content,
                    "type": "example"
                }
                results.append(result)
    return results


def scrape_examples(github_api_token: str | None = None):
    url = "https://api.github.com/repos/IBM/ibm-generative-ai/contents/examples"
    return __scrape_examples_dir(url, github_api_token)


def scrape_all(github_api_token: str | None = None):
    results = []
    results.extend(scrape_documentation(github_api_token))
    results.extend(scrape_examples(github_api_token))
    return results

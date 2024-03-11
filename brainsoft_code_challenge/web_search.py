from collections.abc import Mapping
from typing import Any

import requests
from bs4 import BeautifulSoup
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.runnables import RunnableSerializable
from langchain_openai import ChatOpenAI

from brainsoft_code_challenge.config import WEB_SEARCH_SUMMARIZE_MAX_TOKENS

search = GoogleSerperAPIWrapper()


SUMMARY_TEMPLATE = """{text}

-----------

Using the above text, answer or extract information about the following query:

> {query}

-----------
If the query cannot be answered using the text and no relevant information can be extracted, write a detailed summarization of text instead. Include all factual information, code examples, numbers, stats, etc. if available."""  # noqa: E501
SUMMARY_PROMPT = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)


def serp_api_search(query: str, num_results: int) -> list[str]:
    results = search.results(query)["organic"][:num_results]
    return [r["link"] for r in results]


def scrape_text(url: str) -> str:
    try:
        response = requests.get(url, timeout=5)
        if response.status_code != 200:  # noqa: PLR2004
            return f"Failed to retrieve the webpage: Status code {response.status_code}"
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text(separator=" ", strip=True)  # type: ignore
    except Exception as e:
        return f"Failed to retrieve the webpage: {e}"


def build_web_search_chain(model: str, temperature: float, model_kwargs: Mapping[str, Any]) -> RunnableSerializable:  # type: ignore
    scrape_and_summarize_chain = RunnablePassthrough.assign(
        summary=RunnablePassthrough.assign(text=lambda x: scrape_text(x["url"])[:10000])
        | SUMMARY_PROMPT
        | ChatOpenAI(model=model, max_tokens=WEB_SEARCH_SUMMARIZE_MAX_TOKENS, temperature=temperature, model_kwargs=dict(model_kwargs))
        | StrOutputParser()
    ) | (lambda x: f"URL: {x['url']}\nSUMMARY: {x['summary']}")

    return (
        RunnablePassthrough.assign(urls=lambda x: serp_api_search(x["query"], 3))
        | (lambda x: [{"query": x["query"], "url": url} for url in x["urls"]])
        | scrape_and_summarize_chain.map()
        | (lambda x: "\n\n".join(x))
    )

from collections.abc import Mapping
from typing import Any

import requests
from bs4 import BeautifulSoup
from langchain.agents import tool
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_openai import ChatOpenAI
from pydantic.v1 import BaseModel, Field

from brainsoft_code_challenge.config import (
    N_WEB_SEARCH_RESULTS,
    WEB_SEARCH_MODEL,
    WEB_SEARCH_MODEL_KWARGS,
    WEB_SEARCH_SCRAPING_MAX_RESULT_LENGTH,
    WEB_SEARCH_SCRAPING_TIMEOUT_SECONDS,
    WEB_SEARCH_SUMMARIZE_MAX_TOKENS,
    WEB_SEARCH_TEMPERATURE,
)

search = GoogleSerperAPIWrapper()


SUMMARY_TEMPLATE = """{text}

-----------

Using the above text, answer or extract information about the following query:

> {query}

-----------
If the query cannot be answered using the text and no relevant information can be extracted, write a detailed summarization of text instead. Include all factual information, code examples, numbers, stats, etc. if available."""  # noqa: E501
SUMMARY_PROMPT = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)


def __serp_api_search(query: str, num_results: int) -> list[str]:
    """
    Search Google using SerpAPI.

    :param query: The query to search.
    :param num_results: The number of results to return.
    :return: The top URLs from the search.
    """
    results = search.results(query)["organic"][:num_results]
    return [r["link"] for r in results]


def __scrape_text(url: str) -> str:
    """
    Function to scrape text from a webpage.
    """
    try:
        response = requests.get(url, timeout=WEB_SEARCH_SCRAPING_TIMEOUT_SECONDS)
        if response.status_code != 200:  # noqa: PLR2004
            return f"Failed to retrieve the webpage: Status code {response.status_code}"
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text(separator=" ", strip=True)  # type: ignore
    except Exception as e:
        return f"Failed to retrieve the webpage: {e}"


def build_web_search_chain(model: str, temperature: float, model_kwargs: Mapping[str, Any]) -> RunnableSerializable:  # type: ignore
    """
    Builds a LangChain chain that searches Google and summarizes the top result pages.

    :param model: The OpenAI model to use.
    :param temperature: The temperature to use for the model.
    :param model_kwargs: The model kwargs.
    :return: The LangChain chain.
    """
    scrape_and_summarize_chain = RunnablePassthrough.assign(
        summary=RunnablePassthrough.assign(text=lambda x: __scrape_text(x["url"])[:WEB_SEARCH_SCRAPING_MAX_RESULT_LENGTH])
        | SUMMARY_PROMPT
        | ChatOpenAI(model=model, max_tokens=WEB_SEARCH_SUMMARIZE_MAX_TOKENS, temperature=temperature, model_kwargs=dict(model_kwargs))
        | StrOutputParser()
    ) | (lambda x: f"URL: {x['url']}\nSUMMARY: {x['summary']}")

    return (
        RunnablePassthrough.assign(urls=lambda x: __serp_api_search(x["query"], N_WEB_SEARCH_RESULTS))
        | (lambda x: [{"query": x["query"], "url": url} for url in x["urls"]])
        | scrape_and_summarize_chain.map()
        | (lambda x: "\n\n".join(x))
    )


web_search_chain = build_web_search_chain(WEB_SEARCH_MODEL, WEB_SEARCH_TEMPERATURE, WEB_SEARCH_MODEL_KWARGS)


class GoogleQuery(BaseModel):
    query: str = Field(description="The query to execute")


@tool(args_schema=GoogleQuery)
def search_google(query: str) -> str:
    """Searches Google and returns the summaries of the most relevant results."""  # Tool description for agent
    result = web_search_chain.invoke({"query": query})
    return str(result)

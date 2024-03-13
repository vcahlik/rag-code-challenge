from collections.abc import Mapping
from typing import Any

MODEL_CHOICES = ("gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview")
DEFAULT_MODEL = "gpt-3.5-turbo"
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 1.5
DEFAULT_TEMPERATURE = 0.7
MIN_FREQUENCY_PENALTY = -2.0
MAX_FREQUENCY_PENALTY = 2.0
DEFAULT_FREQUENCY_PENALTY = 0.0
MIN_PRESENCE_PENALTY = -2.0
MAX_PRESENCE_PENALTY = 2.0
DEFAULT_PRESENCE_PENALTY = 0.0
MIN_TOP_P = 0.0
MAX_TOP_P = 1.0
DEFAULT_TOP_P = 0.7

CHROMADB_CHUNK_SIZE = 300  # Number of tokens in each chunk
CHROMADB_CHUNK_OVERLAP = 75  # Number of tokens that each chunk overlaps with the previous one
N_CHROMADB_RESULTS = 15  # This number of chunks is initially returned from ChromaDB (but the document splits may be duplicated)
N_CHROMADB_UNIQUE_RESULTS = 3  # (Up to) this number of unique document splits is returned to the agent

N_WEB_SEARCH_RESULTS = 3  # Number of web search results to return to the agent
WEB_SEARCH_SCRAPING_TIMEOUT_SECONDS = 5  # Maximum time to wait for a web search result to be scraped
WEB_SEARCH_SCRAPING_MAX_RESULT_LENGTH = 10000  # Web search results longer than this (in chars) are truncated
WEB_SEARCH_MODEL = "gpt-3.5-turbo"
WEB_SEARCH_SUMMARIZE_MAX_TOKENS = 1000  # Maximum number of tokens in the summary of a web search result
WEB_SEARCH_TEMPERATURE = 0.7
WEB_SEARCH_MODEL_KWARGS: Mapping[str, Any] = {}

SPLIT_DOCUMENTS_LONGER_THAN_N_CHARS = 8000  # Documentation pages longer than this value are not shown to the agent whole, but are split
MIN_SPLIT_LENGTH_CHARS = 2000  # Minimum length of a document split (which is shown to the agent whole)

CONVERSATION_SUMMARY_MODEL = "gpt-3.5-turbo"  # Model used for summarizing conversations if they exceed memory size

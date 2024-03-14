# Python AI Code Challenge

This repository contains a submission to a RAG agent coding challenge.

## How to Run

### Prerequisites

- Poetry ([installation guide](https://python-poetry.org/docs/))
- Linux operating system (the code has not been tested on Windows or macOS)

### Steps

1. Clone the repository and navigate to it.
2. Copy the file `.env.example` to `.env` and fill in the required environment variables (obtain the free `SERPER_API_KEY` [here](https://serper.dev/) and the free `BEARLY_API_KEY` [here](https://bearly.ai/dashboard/developers), `GITHUB_API_TOKEN` is optional and helps to avoid GitHub API rate limiting).
3. Create a Poetry virtual environment and install the project dependencies:
   ```bash
   poetry install
   ```
4. Activate the newly created virtual environment:
   ```bash
   poetry shell
   ```
5. Scrape the documentation:
   ```bash
   python scripts/scrape_documentation.py
   ```
6. Split long documents into smaller document splits:
   ```bash
   python scripts/split_documentation.py
   ```
7. Chunk the document splits and build the vector database:
   ```bash
   python scripts/rebuild_chromadb.py
   ```
8. To run the Streamlit app:
   ```bash
   streamlit run Assistant.py
   ```
9. To run the CLI version of the assistant:
   ```bash
   python shell_assistant.py
   ```
10. To run the REST API assistant server:
    ```bash
    uvicorn api:app
    ```

## Description of the Design Choices and Result

### Agent

I decided to use OpenAI's chat models and LangChain's *OpenAI Tools agent*, as it is built upon the OpenAI *tools* feature that OpenAI's models have been fine-tuned for. The user can switch between different models (`gpt-3.5-turbo`, `gpt-4`, and `gpt-4-turbo-preview`), but the system prompts have been optimized for `gpt-3.5-turbo` to save API credits. The default temperature and top-p values have been lowered (compared to what OpenAI uses as default) to improve results with RAG search and code interpreter.

For agent memory, I chose `ConversationSummaryBufferMemory` as it allows for long conversations without absolute truncation of the history.

### Documentation search (RAG Pipeline)

For this challenge, I chose *ChromaDB* due to its simplicity (even though it does not offer the most features).

The documentation is scraped from its reStructuredText source files. To ensure that logical blocks of content are returned by the retrieval tool, we do not return chunks, but either whole documents, or "document splits" for long documents (the splits are determined by reStructuredText subsections). To ensure that the text embeddings are accurate, they are calculated chunk-wise (meaning that the same document split can be returned multiple times for a simple query - so a deduplication step is included). I did not implement reranking due to time constraints, but it would certainly benefit the pipeline (e.g. by using *maximal marginal relevance* or a language model for scoring).

### Web Search

I implemented web search using [Serper API](https://serper.dev/), a wrapper around Google Search. I took inspiration from Harrison Chase's implementation of the *GPT Researcher* tool - for a given query, the top three Google search results are scraped and each of them is analyzed by a separate LLM call to either answer the question or summarize the web page contents.

### File Uploading

I implemented file uploading for all versions of the assistant. However, Streamlit's `file_uploader` does not seem to allow automatic resets of its files, so the user must manually de-attach the files after the input has been submitted.

For every file, its text is extracted (I did not implement OCR for scanned PDF files) and prepended before the user input message. The input may be truncated to fit inside the LLM context.

### Code Interpreter

To me, this was the most challenging part of the exercise. As I did not have the ambition to implement a fully sandboxed Python environment, I came across two implementations that seemed reasonable - [Code Interpreter API](https://blog.langchain.dev/code-interpreter-api/) and [Bearly Code Interpreter](https://python.langchain.com/docs/integrations/tools/bearly). However, the Code Interpreter API seems to be currently broken and I haven't yet received support from the developer. Bearly code interpreter is quite limited in its functionality (no support for sessions or installation of new packages), so the Python SDK can't be used inside the tool. I also did not try to implement uploading of custom files, which is not supported ideally by Bearly code interpreter. However, for basic use-cases, the code interpreter tool seems to mostly work.

### Streamlit UI

I chose Streamlit as the framework for UI implementation, as I am familiar with it. I wanted to support better interpretability of the agent behavior, meaning that the user is not only presented with the agent's stream, but can also see the agent tool calls. This seems to work well, however sometimes the output takes a while to refresh right after the user submits a message (I am not absolutely certain whether this can be resolved when using Streamlit).

### CLI

I chose the [Rich](https://github.com/Textualize/rich) library for console outputs (as features such as Markdown rendering are supported out of the box), and the [Prompt Toolkit](https://python-prompt-toolkit.readthedocs.io/en/master/) library for handling user input (with support for going through history). Files can be attached with the `load` command.

### API

I chose REST ([FastAPI](https://fastapi.tiangolo.com/)) for the API, as I am familiar with it (although it does not seem to allow for streaming of outputs). To keep the scope limited, the conversation sessions are not stored by the server, but the user can obtain the conversation history together with every response and pass it with the next request. Files can be uploaded as base64-encoded strings.

## Completion of Objectives

- ✔ Create an application that allows the user to query SDK documentation.
  - ✔️ The agent remembers the conversation context.
    - LangChain's `ConversationSummaryBufferMemory` is used in Streamlit and CLI assistants. In the REST API, message history is returned together with each response, and can be passed with the next request.
  - ✔️ The agent handles large conversations.
    - This is handled by `ConversationSummaryBufferMemory` and by truncation of user inputs that are too long. However, errors can still be encountered when LangChain's `AgentExecutor` itself procudes too long context - this seems to be a current limitation of LangChain and I did not attempt to mitigate it as part of this submission.
  - ✔️ Every response related to the SDK documentation must contain sources (relevant links to the documentation page).
    - This should be handled by the system prompt (a link to the documentation page is returned from the vector database together with each result).
- ✔ Introduce UI for communication with the agent ([Streamlit](https://streamlit.io/))
  - ❔ Handle edge cases and crashes.
    - Most issues have hopefully been resolved and automated `pytest` and LangChain evaluation tests should cover most of the functionality. However, the submission is not perfect - the limitations include a small lag when Streamlit renders messages, errors when the `AgentExectutor` generates too long context (as part of its inner loop, this seems to be a limitation of LangChain), imperfect code interpreter, and truncation of user input when large files are attached.
  - ✔ Use streaming for responses.
    - Implemented for Streamlit and CLI assistants. In Streamlit, tool responses are visualized within the rest of the output stream.
  - ✔ (optional) Allow the user to upload file(s).
    - Implemented for all versions of the assistant.
- ❔ (optional) Evaluation (how your solution performs, how precise it is in terms of retrieval quality).
  - Basic evaluation, which checks that the agent correctly uses its tools, has been implemented in `scripts/evaluate.py`.
- ✔ (optional) The agent can lookup for specific facts on the web (Google / DuckDuckGo).
  - Implemented for all versions of the assistant.
- ❔ (optional) The agent can execute Python code.
  - A simple code interpreter tool is available thanks to [Bearly API](https://python.langchain.com/docs/integrations/tools/bearly). However, this code interpreter does not allow sessions, installation of new packages (including the Python SDK), and even though support for uploading files has not been implemented, the agent still sometimes attempts to do so (at least with `gpt-3.5-turbo`).
- ✔ (optional) The agent can work with files (CSV / PDF).
  - Implemented for all versions of the assistant.
- ✔ (optional) One can interact with the agent via CLI.
  - Rendering of Markdown is supported, as well as prompt history navigation.
- ✔ (optional) One can interact with the agent via API (Rest/gRPC).
  - A REST API has been implemented.

from brainsoft_code_challenge.utils import load_environment

load_environment()

import logging  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
from collections.abc import Mapping  # noqa: E402
from typing import Any  # noqa: E402

from langchain.evaluation import EvaluatorType, load_evaluator  # noqa: E402

from brainsoft_code_challenge.agent import build_agent_input, get_agent_executor  # noqa: E402
from brainsoft_code_challenge.config import DEFAULT_MODEL  # noqa: E402

logging.basicConfig(level=logging.INFO)


EVALUATION_DATASET = [
    {
        "input": "What are the top features of the Python SDK?",
        "reference": """The top features of the IBM Generative AI Python SDK include:

    Very Performant.
    Generated Typings directly from the API.
    Smart Requests Concurrency Handling.
    Retry Mechanism in case of network or API failure.
    Batching Large Requests automatically.
    Easy to extend.
    Integrations to LangChain, LLamaIndex, and HuggingFace.
    LocalServer extension - run a local API compatible with the SDK.

You can find more information on the top features of the Python SDK on the IBM Generative AI Python SDK documentation page.""",
    },
    {
        "input": "How can I determine which version of a given endpoint the SDK uses? I don't need examples. Include the link to documentation.",
        "reference": """To determine which version of a given endpoint the IBM Generative AI Python SDK uses, you can refer to the API Endpoint Versions section in the documentation. The SDK automatically handles versions, and each SDK release is compatible with the latest API version at the time of release. If you need to use the SDK with an older API version, you would need to download a version of the SDK tied to the specific API version you want.

You can access the API Endpoint Versions information in the documentation at the following link: API Endpoint Versions in IBM Generative AI Python SDK Documentation""",  # noqa: E501
    },
    {
        "input": 'Give me the code example "Show information about supported models", exactly as stated in the documentation. A slight modification can be tolerated. Do not modify the example in any way. Produce only code, your output is machine-processed! Do not write any messages!',  # noqa: E501
        "reference": '''from pprint import pprint

from dotenv import load_dotenv

from genai.client import Client
from genai.credentials import Credentials

# make sure you have a .env file under genai root with
# GENAI_KEY=<your-genai-key>
# GENAI_API=<genai-api-endpoint>
load_dotenv()
client = Client(credentials=Credentials.from_env())


def heading(text: str) -> str:
    """Helper function for centering text."""
    return "\n" + f" {text} ".center(80, "=") + "\n"


print(heading("List all models"))
for model in client.model.list(limit=100).results:
    print(model.model_dump(include=["name", "id"]))

print(heading("Get model detail"))
model_detail = client.model.retrieve("google/flan-t5-xl").result
pprint(model_detail.model_dump(include=["name", "description", "id", "developer", "size"]))''',
    },
    {
        "input": "Calculate the first 50 Fibonacci numbers. Use the code interpreter.",
        "reference": "The first 50 Fibonacci numbers are: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025, 121393, 196418, 317811, 514229, 832040, 1346269, 2178309, 3524578, 5702887, 9227465, 14930352, 24157817, 39088169, 63245986, 102334155, 165580141, 267914296, 433494437, 701408733, 1134903170, 1836311903, 2971215073, 4807526976, 7778742049]",  # noqa: E501
    },
    {
        "input": "Ask Google when Python 3.12 was released. Output only the value in the MM-DD-YYYY format and nothing else - your output is machine processed!",  # noqa: E501
        "reference": "10-02-2023",
    },
]


def get_agent_response(input_text: str) -> Mapping[str, Any]:
    agent_executor = get_agent_executor(DEFAULT_MODEL, 0.0, 0.0, 0.0, 1.0, verbose=False)
    agent_input, _ = build_agent_input(input_text, input_files=[], model=DEFAULT_MODEL)
    return agent_executor.invoke(agent_input)


def evaluate() -> list[Any]:
    evaluator = load_evaluator(EvaluatorType.LABELED_CRITERIA, criteria="correctness")
    results = []

    for instance in EVALUATION_DATASET:
        input_text = instance["input"]
        reference_text = instance["reference"]
        prediction_text = get_agent_response(input_text)["output"]
        result = evaluator.evaluate_strings(  # type: ignore
            input=input_text,
            prediction=prediction_text,
            reference=reference_text,
        )
        results.append(result)
        reasoning = result["reasoning"]
        value = result["value"]
        score = result["score"]

        if value == "Y":
            logging.info(f"{input_text}: score {score}")
        else:
            logging.error(f"{input_text}: score {score}, reasoning: {reasoning}")

    failed_results = [result for result in results if result["value"] != "Y"]
    logging.info(f"Average score: {statistics.mean([result['score'] for result in results])}")
    logging.info(f"{len(results)} results evaluated, {len(failed_results)} failures.")
    return failed_results


if __name__ == "__main__":
    success = evaluate()
    if not success:
        sys.exit(1)

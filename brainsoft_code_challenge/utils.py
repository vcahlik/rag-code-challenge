import os

from dotenv import load_dotenv


def load_environment() -> None:
    load_dotenv()


def is_pytest_running() -> bool:
    return "PYTEST_CURRENT_TEST" in os.environ

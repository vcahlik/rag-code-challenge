import os

from dotenv import load_dotenv


def load_environment() -> None:
    """
    Load the environment variables from the .env file.
    """
    load_dotenv()


def is_pytest_running() -> bool:
    """
    Check if the code is running in a pytest environment.
    """
    return "PYTEST_CURRENT_TEST" in os.environ

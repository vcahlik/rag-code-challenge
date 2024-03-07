import os

from dotenv import load_dotenv


def load_environment():
    load_dotenv()


def is_pytest_running():
    return "PYTEST_CURRENT_TEST" in os.environ

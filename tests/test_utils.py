from brainsoft_code_challenge.utils import is_pytest_running


def test_pytest_is_running():
    assert is_pytest_running() is True  # noqa: S101

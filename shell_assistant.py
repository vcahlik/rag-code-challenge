from brainsoft_code_challenge.utils import load_environment

load_environment()

import asyncio  # noqa: E402
import sys  # noqa: E402

from brainsoft_code_challenge.agent import get_agent_executor  # noqa: E402


async def run():
    agent_executor = get_agent_executor(verbose=False)
    print("How can I help you?")
    while True:
        user_input = input("User: ")
        sys.stdout.write("Assistant: ")
        async for event in agent_executor.astream_events({"input": user_input}, version="v1"):
            if event["event"] == "on_chat_model_stream":
                sys.stdout.write(event["data"]["chunk"].content)
                sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()


if __name__ == "__main__":
    asyncio.run(run())

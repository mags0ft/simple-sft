"""
Efficiently interfaces with cloud, remote or locally running LLMs by managing
large amounts of tasks in parallel for a more efficient use of time.

Because of no widely available support for batch processing across providers,
requests are still handled individually. This may change in the future.
"""

import json
from os import getenv
from threading import Thread
import time
from typing import Any

from dotenv import load_dotenv
import openai

from config_reader import config
from constants import OPENAI_API_KEY_ENV, OPENAI_BASE_URL_ENV
from custom_types import ToolCallType


load_dotenv()

client = openai.OpenAI(
    base_url=getenv(OPENAI_BASE_URL_ENV), api_key=getenv(OPENAI_API_KEY_ENV)
)

base_config = {
    "model": config["model"],
    "max_tokens": config["api_query"]["max_tokens"],
    "reasoning_effort": config["api_query"]["reasoning_effort"],
    "extra_body": (
        json.loads(config["api_query"]["extra_body"])
        if config["api_query"]["extra_body"]
        else {}
    ),
    "stream": False,
    "extra_headers": {
        "X-OpenRouter-Title": "simple-sft",
        "X-OpenRouter-Categories": "general-chat",
        "HTTP-Referer": "https://github.com/mags0ft/simple-sft",
    },
}

meta_config = {
    "temperature": config["api_query"]["meta_temperature"],
    **base_config,
}
chat_config = {
    "temperature": config["api_query"]["chat_temperature"],
    **base_config,
}


class OpenAIAPIRequestError(Exception):
    """
    Error raised when there is no response available.
    """


def completion_wrapper(
    **kwargs,
) -> "Any":
    wait_amount = 2

    for _ in range(config["api_query"]["max_retries"]):
        response = client.chat.completions.create(**kwargs)

        if (
            response.status_code == 200
            and response.choices[0].message
            and response.choices[0].message.content
        ):
            return response

        # We do this to fix possible rate limits
        time.sleep(wait_amount)
        wait_amount *= 2

    raise OpenAIAPIRequestError(f"Failed to get a valid response.")


def _check_response(response: Any | None) -> None:
    """
    Checks if a response seems to be roughly valid.
    """

    if response is None:
        raise ValueError("Empty response (is None)")

    if not response.choices:
        raise ValueError("No choices in the response avaiable (len 0).")


def get_text(response: Any | None) -> str:
    """
    Extracts the pure text from the response object given.
    """

    _check_response(response)
    text_content: str = response.choices[0].message.content  # type: ignore

    return text_content.strip()


def get_reasoning(response: Any | None) -> str:
    """
    Extracts the reasoning traces from the response object given.
    """

    _check_response(response)
    reasoning_content = getattr(response, "reasoning", "")

    return reasoning_content.strip()


def get_tool_calls(response: Any | None) -> list[ToolCallType]:
    """
    Retrieves the tool calls from a specific response.
    """

    _check_response(response)

    raw_tool_calls = response.choices[0].message.tool_calls  # type: ignore

    if not raw_tool_calls:
        return []

    tool_calls: list[ToolCallType] = []

    for tool_call in raw_tool_calls:
        tool_id = tool_call.id
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)

        tool_calls.append(
            {
                "type": "function",
                "id": tool_id,
                "function": {"name": tool_name, "arguments": tool_args},
            }
        )

    return tool_calls


def simple_in_out(input_: str) -> str:
    """
    Processes a query on the OpenAI API compatible endpoint and returns the
    plain response text with no reasoning traces. No streaming, no background
    mode.
    """

    response = completion_wrapper(
        messages=[{"role": "user", "content": input_.strip()}],
        **meta_config,
    )

    return get_text(response)


def process_many_out_of_order(prompts: list[str], n_threads: int = 8) -> list[str]:
    """
    Processes many given prompts out-of-order and returns a list of the
    resulting texts without reasoning traces.
    """

    results: list[str] = []

    def worker(prompt: str) -> None:
        results.append(simple_in_out(prompt))

    threads = []

    for prompt in prompts:
        thread = Thread(target=worker, args=(prompt,))
        thread.start()
        threads.append(thread)

        # Limit amount of threads:
        while len(threads) >= n_threads:
            threads = [t for t in threads if t.is_alive()]

    # Wait for all to finish
    for thread in threads:
        thread.join()

    return results

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
from logging_manager import logger


load_dotenv()

client = openai.OpenAI(
    base_url=getenv(OPENAI_BASE_URL_ENV), api_key=getenv(OPENAI_API_KEY_ENV)
)
logger.debug("OpenAI client configured (base_url=%s)", getenv(OPENAI_BASE_URL_ENV))

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
    for attempt in range(config["api_query"]["max_retries"]):
        logger.debug("LLM request attempt %d", attempt + 1)
        response = client.chat.completions.create(**kwargs)

        if response.choices[0].message and response.choices[0].message.content:
            logger.debug("LLM request succeeded on attempt %d", attempt + 1)
            return response

        logger.warning(
            "LLM request returned non-OK or empty on attempt %d, retrying", attempt + 1
        )
        # We do this to fix possible rate limits
        time.sleep(wait_amount)
        wait_amount *= 2

    logger.error(
        "LLM request failed after %d attempts", config["api_query"]["max_retries"]
    )
    raise OpenAIAPIRequestError(f"Failed to get a valid response.")


def _check_response(response: Any | None) -> None:
    """
    Checks if a response seems to be roughly valid.
    """

    if response is None:
        logger.error("Empty response (is None)")
        raise ValueError("Empty response (is None)")

    if not response.choices:
        raise ValueError("No choices in the response avaiable (len 0).")


def get_text(response: Any | None) -> str:
    """
    Extracts the pure text from the response object given.
    """

    _check_response(response)
    text_content: str = response.choices[0].message.content  # type: ignore
    logger.debug("Extracted text content of length %d", len(text_content or ""))

    return text_content.strip()


def get_reasoning(response: Any | None) -> str:
    """
    Extracts the reasoning traces from the response object given.
    """

    _check_response(response)
    reasoning_content = getattr(response, "reasoning", "")

    if not reasoning_content:
        try:
            reasoning_content = response.choices[0].message.reasoning_details.text  # type: ignore
        except AttributeError:
            reasoning_content = ""
    logger.debug("Extracted reasoning content length %d", len(reasoning_content or ""))
    return reasoning_content.strip()


def get_tool_calls(response: Any | None) -> list[ToolCallType]:
    """
    Retrieves the tool calls from a specific response.
    """

    _check_response(response)

    raw_tool_calls = response.choices[0].message.tool_calls  # type: ignore

    if not raw_tool_calls:
        logger.debug("No tool calls found in response")
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

    logger.debug("Sending simple_in_out prompt (len=%d)", len(input_ or ""))
    response = completion_wrapper(
        messages=[{"role": "user", "content": input_.strip()}],
        **meta_config,
    )

    text = get_text(response)
    logger.debug("simple_in_out received text of length %d", len(text))
    return text


def process_many_out_of_order(prompts: list[str], n_threads: int = 8) -> list[str]:
    """
    Processes many given prompts out-of-order and returns a list of the
    resulting texts without reasoning traces.
    """

    results: list[str] = []

    def worker(prompt: str) -> None:
        logger.debug("Worker thread sending prompt (len=%d)", len(prompt or ""))
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

    logger.debug("process_many_out_of_order completed (%d results)", len(results))
    return results


def retrieve_several_as_structured_output(
    prompt: str, resp_json_array_name: str = "messages"
) -> list[str]:
    """
    Retrieves several outputs for a given prompt in parallel and returns them
    as a list of strings.

    Uses the OpenAI API's structured_output feature to get the outputs in a
    JSON array format.
    """

    logger.debug("Retrieving structured output for prompt (len=%d)", len(prompt or ""))
    response = completion_wrapper(
        messages=[{"role": "user", "content": prompt.strip()}],
        response_format={
            "type": "json_object",
            "properties": {
                resp_json_array_name: {
                    "type": "array",
                    "items": {"type": "string"},
                }
            },
            "required": [resp_json_array_name],
        },
        **meta_config,
    )

    response_text = get_text(response)
    logger.debug("Structured response text length %d", len(response_text or ""))

    try:
        response_json = json.loads(response_text)

        if isinstance(response_json, dict):
            return response_json[resp_json_array_name]
        elif isinstance(response_json, list):
            logger.debug("Response JSON is a list, returning it directly")
            return response_json
        else:
            logger.error("Unexpected JSON structure: %s", type(response_json).__name__)
            raise ValueError(
                f"Unexpected JSON structure: {type(response_json).__name__}"
            )
    except (json.JSONDecodeError, KeyError) as e:
        logger.error("Failed to parse structured response: %s", e)
        raise ValueError(
            f"Failed to parse the response as JSON or expected key not found: {e}"
        )

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

from dotenv import load_dotenv
import openai

from config_reader import config
from constants import OPENAI_API_KEY_ENV, OPENAI_BASE_URL_ENV


load_dotenv()

client = openai.OpenAI(
    base_url=getenv(OPENAI_BASE_URL_ENV), api_key=getenv(OPENAI_API_KEY_ENV)
)


base_config = {
    "model": config["model"],
    "max_tokens": config["api_query"]["max_tokens"],
    "reasoning_effort": config["api_query"]["reasoning_effort"],
    "extra_body": json.loads(config["api_query"]["extra_body"]),
    "stream": False,
    "extra_headers": {
        "X-OpenRouter-Title": "simple-sft",
        "X-OpenRouter-Categories": "general-chat",
        "HTTP-Referer": "",
    },
}

meta_config = {"temperature": config["api_query"]["meta_temperature"], **base_config}
chat_config = {"temperature": config["api_query"]["chat_temperature"], **base_config}


class OpenAIAPIRequestError(Exception):
    """
    Error raised when there is no response available.
    """


def completion_wrapper(**kwargs) -> openai.ChatCompletion:
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

    raise


def get_text(response: openai.ChatCompletion | None) -> str:
    """
    Extracts the pure text from the response object given.
    """

    if response is None:
        raise ValueError("Empty response (is None)")

    if not response.choices:
        raise ValueError("No choices in the response avaiable (len 0).")

    text_content: str = response.choices[0].message.content

    return text_content.strip()


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

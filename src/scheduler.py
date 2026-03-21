"""
Schedules the execution of n threads in parallel and correctly writes their
responses atomically into an output JSONL file. Handles resuming after a crash.
"""

import json
import random
import threading
import os
import uuid

from dataclasses import dataclass, field

from config_reader import config
from llm_interface import retrieve_several_as_structured_output
from prompts import (
    SYSTEM_PROMPT_GENERATION_ADDITION,
    SYSTEM_PROMPT_GENERATION_PROMPT,
    concatenate_prompts,
)

from logging_manager import logger


_JSONL_WRITE_LOCK = threading.Lock()


@dataclass
class Category:
    """
    Stores meta information about a category.
    """

    n_rows_total: int = 0
    n_rows_per_language: dict[str, int] = field(default_factory=dict)
    name: str = ""


def initialize_output_directory(run_name: str) -> tuple[str, str]:
    """
    Initializes the output directory for the given run name.
    """

    RUN_NAME = run_name + "_" + str(uuid.uuid4())[:8]

    os.makedirs(os.path.join("data", RUN_NAME), exist_ok=True)
    logger.debug("Initialized output directory: %s", RUN_NAME)

    # create system_prompts.jsonl and conversations.jsonl files
    system_prompts_path = os.path.join("data", RUN_NAME, f"system_prompts.jsonl")
    conversations_path = os.path.join("data", RUN_NAME, f"conversations.jsonl")

    with open(system_prompts_path, "w") as f:
        pass

    with open(conversations_path, "w") as f:
        pass

    return system_prompts_path, conversations_path


def write_atomically_to_jsonl(file_path: str, data: list[dict]) -> None:
    """
    Using the threading library, writes the given data atomically to the given
    file path in JSONL format.
    """

    with _JSONL_WRITE_LOCK:
        logger.debug("Writing %d items to %s", len(data), file_path)
        with open(file_path, "a") as file_handle:
            file_handle.write(
                "\n".join([json.dumps(item, ensure_ascii=False) for item in data])
            )
            if data:
                file_handle.write("\n")
        logger.debug("Wrote %d items to %s", len(data), file_path)


def generate_system_prompts_in_parallel(output_file_path: str) -> None:
    """
    Generates system prompts in parallel using multiple threads. Needs to store
    the generated system prompts in the output_file_path atomically across
    threads.
    """

    themes = config["prompting"]["system_prompt_generation_themes"]
    formats = config["prompting"]["system_prompt_formats"]
    n_system_prompts = config["prompting"]["n_system_prompts"]
    n_threads = config["n_threads"]

    assert isinstance(n_system_prompts, int) and n_system_prompts > 0
    assert isinstance(n_threads, int) and n_threads > 0

    def worker(n_prompts: int):
        """
        The worker function for generating system prompts. Each thread runs
        this function with its own theme and format.
        """

        is_special = (
            config["prompting"]["include_special_system_prompts"]
            and random.random()
            < config["prompting"]["special_system_prompts_percentage"]
        )

        left_to_generate = n_prompts

        logger.debug(
            "Thread %s: starting to generate %d prompts (special=%s)",
            threading.get_ident(),
            n_prompts,
            is_special,
        )

        while left_to_generate > 0:
            theme = random.choice(themes)
            format_ = random.choice(formats)

            base_prompt = SYSTEM_PROMPT_GENERATION_PROMPT % (theme, format_)
            input_prompt = (
                base_prompt
                if not is_special
                else concatenate_prompts(base_prompt, SYSTEM_PROMPT_GENERATION_ADDITION)
            )

            prompts = retrieve_several_as_structured_output(
                input_prompt,
                resp_json_array_name="prompts",
            )

            if not prompts:
                logger.error(
                    "Thread %s: Failed to generate system prompts.",
                    threading.get_ident(),
                )
                raise ValueError("Failed to generate system prompts.")

            prompts_to_write = prompts[:left_to_generate]

            left_to_generate -= len(prompts_to_write)

            # write prompts atomically to file:
            write_atomically_to_jsonl(
                output_file_path,
                [{"prompt": prompt} for prompt in prompts_to_write],
            )

        logger.debug("Thread %s: finished generating prompts", threading.get_ident())

    if n_system_prompts == 0:
        logger.info("No system prompts requested (n_system_prompts == 0)")
        return

    thread_count = min(n_threads, n_system_prompts)
    base_prompts_per_thread, remainder = divmod(n_system_prompts, thread_count)

    threads = []

    for thread_index in range(thread_count):
        prompts_for_thread = base_prompts_per_thread
        if thread_index < remainder:
            prompts_for_thread += 1

        thread = threading.Thread(target=worker, args=(prompts_for_thread,))
        thread.start()
        logger.debug(
            "Started system prompt thread %d (id=%s) with %d prompts",
            thread_index,
            thread.ident,
            prompts_for_thread,
        )
        threads.append(thread)

    for thread in threads:
        thread.join()
    logger.info("All system prompt threads joined")


def generate_conversations_in_parallel(
    output_file_path: str, system_prompts_path: str
) -> None:
    """
    Generates conversations in parallel using multiple threads. Re-uses the
    previously generated system prompts from the system_prompts_path. Stores
    the generated conversations in the output_file_path atomically across
    threads.
    """


def calculate_per_language(
    total_rows: int, languages: dict[str, float]
) -> tuple[dict[str, int], int]:
    """
    Calculates the number of rows for each language based on their percentages.
    Returns the number of rows for each language and the minimum total number
    of rows across all languages.
    """

    results = {}
    total_percent = sum(percent for percent in languages.values())

    for lang, percent in languages.items():
        results[lang] = max(
            int(total_rows * percent / total_percent), config["batch_size"]
        )

    min_total_rows = sum(results.values())

    return results, min_total_rows


def distribute_categories(
    total_rows: int, categories: dict[str, float | int]
) -> list[Category]:
    """
    Distributes the total number of rows across different categories based on
    their weights.
    """

    results = []

    rows_left = total_rows

    # first, handle all categories with a fixed number of rows:
    for category_name, weight in categories.items():
        if isinstance(weight, int) and weight >= 0:
            language_distribution, total_rows_for_category = calculate_per_language(
                weight, config["languages"]
            )

            results.append(
                Category(
                    n_rows_total=total_rows_for_category,
                    n_rows_per_language=language_distribution,
                    name=category_name,
                )
            )

            # we do not subtract by total_rows_for_category here, because the
            # user shall not need to care about the batch_size when writing the
            # config
            rows_left -= weight

        assert (
            rows_left >= 0
        ), "Total number of rows is less than the sum of fixed rows."

    # now, let's see how many percent the remaining categories have in total:
    total_percent = sum(
        weight for weight in categories.values() if isinstance(weight, float)
    )

    # let's count how many categories are -1 ("go figure it out for me"):
    n_negative_one = sum(1 for weight in categories.values() if weight == -1)

    # we do this in order to evenly distribute the -1 weights across the
    # remaining categories:
    for category_name, weight in categories.items():
        if weight == -1:
            categories[category_name] = (1 - total_percent) / n_negative_one

    # now we can distribute the remaining rows according to the percentage
    # weights:
    for category_name, weight in categories.items():
        if isinstance(weight, float):
            n_rows_for_category = int(total_rows * weight)

            language_distribution, actual_rows_for_category = calculate_per_language(
                n_rows_for_category, config["languages"]
            )

            results.append(
                Category(
                    n_rows_total=actual_rows_for_category,
                    n_rows_per_language=language_distribution,
                    name=category_name,
                )
            )

    return results


def main_flow():
    """
    Main flow of the program. Needs to handle a lot: Orchestrating the topic,
    language and system prompt distribution among conversations, correctly
    batching the generated initial prompts and system prompts etc.
    """

    n_threads = config["n_threads"]
    n_rows = config["rows"]

    logger.debug("Main flow started: n_threads=%d, n_rows=%d", n_threads, n_rows)

    assert isinstance(n_threads, int) and n_threads > 0
    assert isinstance(n_rows, int) and n_rows > 0

    distribution = distribute_categories(n_rows, config["categories"])

    system_prompts_path, conversations_path = initialize_output_directory(
        config["run_name"]
    )

    generate_system_prompts_in_parallel(system_prompts_path)
    # generate_conversations_in_parallel(conversations_path, system_prompts_path)

"""
Schedules the execution of n threads in parallel and correctly writes their
responses atomically into an output JSONL file. Handles resuming after a crash.
"""

import threading
import os
import uuid
from config_reader import config


def initialize_output_directory(run_name: str) -> tuple[str, str]:
    """
    Initializes the output directory for the given run name.
    """

    RUN_NAME = run_name + "_" + str(uuid.uuid4())[:8]

    os.makedirs(os.path.join("data", RUN_NAME), exist_ok=True)

    # create system_prompts.jsonl and conversations.jsonl files
    system_prompts_path = os.path.join("data", RUN_NAME, f"system_prompts.jsonl")
    conversations_path = os.path.join("data", RUN_NAME, f"conversations.jsonl")

    with open(system_prompts_path, "w") as f:
        pass

    with open(conversations_path, "w") as f:
        pass

    return system_prompts_path, conversations_path


def generate_system_prompts_in_parallel(output_file_path: str) -> None:
    """
    Generates system prompts in parallel using multiple threads.
    """


def generate_conversations_in_parallel(output_file_path: str) -> None:
    """
    Generates conversations in parallel using multiple threads.
    """


def main_flow():
    """
    Main flow of the program.
    """

    system_prompts_path, conversations_path = initialize_output_directory(
        config["run_name"]
    )

    generate_system_prompts_in_parallel(system_prompts_path)
    generate_conversations_in_parallel(conversations_path)

"""
Handles the generation of many system prompts for the conversations generated.
"""

from random import choice

from llm_interface import process_many_out_of_order
from prompts import SYSTEM_PROMPT_GENERATION_ADDITION, SYSTEM_PROMPT_GENERATION_PROMPT


def generate_system_prompts(
    amount: int,
    special_prompts: bool,
    themes: list[str],
    formats: list[str],
    special_prompts_percentage: float = 0.0,
) -> list[str]:
    """
    Generates a batch of versatile system prompts.
    """

    assert not special_prompts or special_prompts_percentage == 0.0
    assert special_prompts_percentage < 1.0

    n_normal_prompts = (
        int(amount * (1 - special_prompts_percentage)) if special_prompts else amount
    )

    n_special_prompts = (
        int(amount * special_prompts_percentage) if special_prompts else 0
    )

    generation_prompts = [
        SYSTEM_PROMPT_GENERATION_PROMPT % (choice(themes), choice(formats))
        for _ in range(n_normal_prompts)
    ] + [
        (SYSTEM_PROMPT_GENERATION_PROMPT + " " + SYSTEM_PROMPT_GENERATION_ADDITION)
        % (choice(themes), choice(formats))
        for _ in range(n_special_prompts)
    ]

    return process_many_out_of_order(generation_prompts)

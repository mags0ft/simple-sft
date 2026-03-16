"""
Handles the main secret sauce: The conversation generation.
"""

from random import random
from uuid import uuid4

from config_reader import config
from llm_interface import simple_in_out


MessagesType = list[dict[str, str | dict]]
ConversationType = dict[str, str | MessagesType]


def generate_conversation(
    category: str, system_prompt: str, initial_question: str
) -> ConversationType:
    """
    Generates a conversation.
    """

    assert category.strip() != ""
    assert not config["output"]["add_system_prompts"] or system_prompt.strip() != ""

    conversation: ConversationType = {
        "id": uuid4()[:8],
        "messages": [],
        "category": str,
    }

    if config["output"]["add_system_prompts"]:
        conversation["messages"].append({"role": "system", "content": system_prompt})

    turns = 0

    while turns < config["conversation"]["max_length"]:
        user_message = (
            generate_user_message(conversation["messages"], category)
            if turns > 0
            else initial_question
        )
        conversation.append({"role": "user", "content": user_message})

        assistant_message, reasoning, tool_calls = generate_assistant_response(
            conversation
        )
        conversation.append(
            {
                "role": "assistant",
                "content": assistant_message,
                "thinking": reasoning,
                "tool_calls": tool_calls,
            }
        )

        turns += 1

        if (
            turns > config["conversation"]["min_length"]
            and random() > config["conversation"]["extend_prob"]
        ):
            break

    return post_processing(conversation)


def generate_user_message(messages: MessagesType, category: str) -> str:
    """
    Generates a follow-up request or message the user could send to the
    assistant.
    """


def post_processing(conversation: ConversationType) -> ConversationType:
    """
    Applies changes to the final conversation object to make it adhere to the
    user's configuration.
    """

    for message in conversation["messages"]:
        if not config["output"]["include_reasoning_traces"]:
            del message["thinking"]

        elif config["output"]["output_reasoning_field_name"] != "thinking":
            temp = message["thinking"]
            del message["thinking"]
            message[config["output"]["output_reasoning_field_name"]] = temp

    return conversation

"""
Handles the main secret sauce: The conversation generation.
"""

from random import random
from uuid import uuid4

from config_reader import config
from llm_interface import (
    get_reasoning,
    get_tool_calls,
    simple_in_out,
    completion_wrapper,
    get_text,
    chat_config,
)
from prompts import (
    FOLLOWUP_QUESTION_GENERATION_PROMPT,
    concatenate_prompts,
    InjectedSpecialPrompts,
)
from custom_types import ConversationType, MessagesType, ToolCallType, TopLevelToolType
from tools import get_tool_response


class RepetitionError(Exception):
    """
    Error raised when a model called too many tools consecutively.
    """


def generate_conversation(
    category: str,
    system_prompt: str,
    special_category: str,
    initial_question: str,
    language: str,
    tools: list[TopLevelToolType],
) -> ConversationType:
    """
    Generates a conversation.
    """

    assert category.strip() != ""
    assert not config["output"]["add_system_prompts"] or system_prompt.strip() != ""

    conversation: ConversationType = {
        "id": str(uuid4())[:8],
        "messages": [],
        "tools": tools,
        "category": category,
        "specials": special_category if special_category else "none",
    }

    if config["output"]["add_system_prompts"]:
        conversation["messages"].append({"role": "system", "content": system_prompt})

    turns = 0

    while turns < config["conversation"]["max_length"]:
        user_message = (
            generate_user_message(conversation["messages"], category, language)
            if turns > 0
            else initial_question
        )
        conversation["messages"].append({"role": "user", "content": user_message})

        # Handle assistant responses with tool calls:

        for _ in range(config["conversation"]["max_consecutive_tool_calls"]):
            assistant_message, reasoning, tool_calls = generate_assistant_response(
                conversation, language
            )
            conversation["messages"].append(
                {
                    "role": "assistant",
                    "content": assistant_message,
                    "thinking": reasoning,
                    "tool_calls": tool_calls,
                }
            )

            if not tool_calls:
                break

            for tool_call in tool_calls:
                if tool_call["function"]["name"] not in [
                    tool["function"]["name"] for tool in conversation["tools"]
                ]:
                    raise ValueError(
                        f'Model called non-existing tool \
"{tool_call["function"]["name"]}" in the scope of this conversation.'
                    )

                conversation["messages"].append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": get_tool_response(
                            tool_call["function"]["name"],
                            tool_call["function"]["arguments"],
                        ),
                    }
                )
        else:
            raise RepetitionError(
                "Model called tools too often in a row (configure in \
conversation.max_consecutive_tool_calls) without letting the user talk."
            )

        turns += 1

        if (
            turns > config["conversation"]["min_length"]
            and random() > config["conversation"]["extend_prob"]
        ):
            break

    return post_processing(conversation)


def generate_user_message(
    messages: list[MessagesType], category: str, language: str
) -> str:
    """
    Generates a follow-up request or message the user could send to the
    assistant.
    """

    def shorten_too_long_message(message: str) -> str:
        if len(message) > 512:
            return message[:512] + "... [truncated message]"

        return message

    constructed_summary = ""

    for message in messages:
        if message["role"] not in ["user", "assistant"]:
            continue

        constructed_summary += (
            ("**User**: " if message["role"] == "user" else "**Assistant**: ")
            + shorten_too_long_message(message["content"]).strip()
            + "\n\n"
        )

    return simple_in_out(
        FOLLOWUP_QUESTION_GENERATION_PROMPT % (category, constructed_summary, language)
    )


def generate_assistant_response(
    conversation: ConversationType,
    language: str,
) -> tuple[str, str, list[ToolCallType]]:
    """
    Generates the response of the assistant.
    """

    def inject_special_prompt_into_system_prompt(
        messages: list[MessagesType],
        special_category: str = "",
        backstage_instruction: str = "",
        language: str = "English",
    ) -> list[MessagesType]:
        """
        Injects the special instructions (backstage instructions, special
        category warnings) into the model's system prompt. Never seen in the
        final dataset.
        """

        if not messages or messages[0]["role"] != "system":
            return messages

        special_prompt = {
            "prompt_injection": InjectedSpecialPrompts.prompt_injection_warning,
            "hallucination": InjectedSpecialPrompts.hallucination_warning,
            "nonsense": InjectedSpecialPrompts.nonsense_warning,
        }.get(special_category, "")

        messages[0]["content"] = concatenate_prompts(
            messages[0]["content"],
            special_prompt,
            backstage_instruction,
            "Respond in %s." % language,
        )

        return messages

    response = completion_wrapper(
        **chat_config,
        messages=inject_special_prompt_into_system_prompt(
            conversation["messages"], conversation["specials"], language
        ),
        tools=conversation["tools"],
    )

    message = get_text(response)
    reasoning = get_reasoning(response)
    tool_calls = get_tool_calls(response)

    return (message, reasoning or "", tool_calls or [])


def post_processing(conversation: ConversationType) -> ConversationType:
    """
    Applies changes to the final conversation object to make it adhere to the
    user's configuration.
    """

    for message in conversation["messages"]:
        if not config["output"]["include_reasoning_traces"] and "thinking" in message:
            del message["thinking"]

        elif (
            config["output"]["output_reasoning_field_name"] != "thinking"
            and "thinking" in message
        ):
            temp = message["thinking"]
            del message["thinking"]
            message[config["output"]["output_reasoning_field_name"]] = temp

    return conversation

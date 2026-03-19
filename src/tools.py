"""
Definitions and behaviors for tools.
"""

import datetime
import json
import random

from calculator_sandbox import sandboxed_calculator_tool
from llm_interface import simple_in_out
from prompts import WEB_SEARCH_SIMULATION_PROMPT
from webpage_fetcher import fetch_webpage_content_tool


class ToolError(Exception):
    """
    Exception raised when there is an issue with running a tool.
    """


class Tool:
    """
    Generic abstraction for a tool.
    """

    possible_names = ["Generic Tool"]
    possible_descriptions = ["This is a generic tool that does nothing."]
    schema = {}
    callback = lambda _: 0

    def __init__(self, possible_names, possible_descriptions, schema, callback):
        self.possible_names = possible_names
        self.possible_descriptions = possible_descriptions
        self.schema = schema
        self.callback = callback

    def generate_variation(self) -> str:
        """
        Generates a random variation of the tool.
        """

        params = {"type": "object"}
        params.update(self.schema)

        return json.dumps(
            {
                "type": "function",
                "function": {
                    "name": random.choice(self.possible_names),
                    "description": random.choice(self.possible_descriptions),
                    "parameters": params,
                },
            }
        )


# ------------ Callbacks ------------


def _tool_weather(args: dict[str, str]):
    """
    Simulates a weather tool.
    """

    if not args or list(args.keys())[0] not in [
        "location",
        "city",
        "place",
    ]:
        raise ValueError("Invalid arguments for weather tool.")

    weather_report = {
        "location": args.get("location", args.get("city", args.get("place"))),
        "units": "Celsius",
        "curently": {
            "temperature": random.randint(-10, 40),
            "condition": random.choice(
                ["Sunny", "Cloudy", "Rainy", "Snowy", "Windy", "Stormy"]
            ),
        },
        "forecast": [
            {
                "day": datetime.date.today() + datetime.timedelta(days=day),
                "temperature": random.randint(-10, 40),
                "condition": random.choice(
                    ["Sunny", "Cloudy", "Rainy", "Snowy", "Windy", "Stormy"]
                ),
            }
            for day in range(1, 7)
        ],
    }

    return weather_report


def _tool_web_search(args: dict[str, str]) -> dict:
    """
    Simulates the web search tool.
    """

    for _ in range(3):
        try:
            return json.loads(
                simple_in_out(WEB_SEARCH_SIMULATION_PROMPT % args.get("query", ""))
            )
        except json.JSONDecodeError:
            continue

    raise ValueError("Model did not return valid JSON for the web search tool.")


def _tool_calculator(args: dict[str, str]) -> dict:
    """
    Calls the sandboxed calculator tool to evaluate a mathematical expression
    safely.
    """

    return sandboxed_calculator_tool(args)


def _tool_fetch_webpage(args: dict[str, str]) -> dict:
    """
    Fetches a webpage using requests and extracts the text content only using
    BeautifulSoup.
    """

    return fetch_webpage_content_tool(args)


# ----------- Definitions -----------

TOOLS = {
    "weather": Tool([], [], {}, _tool_weather),
    "web_search": Tool([], [], {}, _tool_web_search),
    "calculator": Tool([], [], {}, _tool_calculator),
    "fetch_webpage": Tool([], [], {}, _tool_fetch_webpage),
}


def get_tool_response(name: str, args: str) -> str:
    """
    Retrieves a response for the
    """

    try:
        parsed_args = json.loads(args)
    except json.JSONDecodeError:
        raise ValueError("Model did not return valid JSON for the tool call.")

    try:
        return json.dumps(TOOLS[name].callback(parsed_args))
    except (KeyError, ValueError) as e:
        raise ToolError("The tool did not run successfully: ", e)

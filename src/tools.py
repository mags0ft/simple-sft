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
from logging_manager import logger


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

    logger.debug("Weather tool called with args: %s", args)

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

    logger.debug("Web search tool called with query: %s", args.get("query", ""))
    for attempt in range(3):
        try:
            result = json.loads(
                simple_in_out(WEB_SEARCH_SIMULATION_PROMPT % args.get("query", ""))
            )
            logger.debug("Web search tool succeeded on attempt %d", attempt + 1)
            return result
        except json.JSONDecodeError:
            logger.warning(
                "Web search tool returned invalid JSON on attempt %d", attempt + 1
            )
            continue

    logger.error("Web search tool failed to return valid JSON after 3 attempts")
    raise ValueError("Model did not return valid JSON for the web search tool.")


def _tool_calculator(args: dict[str, str]) -> dict:
    """
    Calls the sandboxed calculator tool to evaluate a mathematical expression
    safely.
    """

    logger.debug("Calculator tool called with args: %s", args)
    return sandboxed_calculator_tool(args)


def _tool_fetch_webpage(args: dict[str, str]) -> dict:
    """
    Fetches a webpage using requests and extracts the text content only using
    BeautifulSoup.
    """

    logger.debug("Fetch webpage tool called with args: %s", args)
    return fetch_webpage_content_tool(args)


# ----------- Definitions -----------

TOOLS = {
    "weather": Tool(
        [
            "get_weather",
            "weather_conditions",
            "weather_report",
            "local_weather",
            "check_weather",
        ],
        [
            "Retrieve current weather conditions and forecast for a specified location",
            "Get the weather forecast and current conditions for a city",
            "Look up temperature, conditions, and 6-day forecast for a location",
            "Check current weather and upcoming forecast for a place",
            "Fetch meteorological data for a geographic location",
        ],
        {
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City or location name to get weather for",
                }
            },
            "required": ["location"],
        },
        _tool_weather,
    ),
    "web_search": Tool(
        [
            "search_web",
            "internet_search",
            "query_engine",
            "find_information",
            "web_query",
        ],
        [
            "Search the internet for information related to a query",
            "Perform a web search to find relevant information and sources",
            "Query search engines to retrieve results and summaries for a topic",
            "Search online resources for information matching your query",
            "Look up information from the web using search queries",
        ],
        {
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to find relevant information",
                }
            },
            "required": ["query"],
        },
        _tool_web_search,
    ),
    "calculator": Tool(
        [
            "calculate",
            "math_eval",
            "compute_expression",
            "evaluate_math",
            "mathematical_solver",
        ],
        [
            "Evaluate a mathematical expression and return the result",
            "Compute the result of a mathematical calculation",
            "Solve a mathematical expression using safe arithmetic operations",
            "Execute mathematical calculations with support for common functions",
            "Calculate the value of a mathematical expression",
        ],
        {
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate \
(supports +, -, *, /, ^, sqrt, sin, cos, tan, log, exp, parentheses)",
                }
            },
            "required": ["expression"],
        },
        _tool_calculator,
    ),
    "fetch_webpage": Tool(
        [
            "fetch_page",
            "get_webpage",
            "extract_content",
            "read_webpage",
            "scrape_webpage",
        ],
        [
            "Fetch a webpage and extract its text content",
            "Retrieve the textual content from a URL",
            "Download and parse a webpage to get readable text content",
            "Extract text content from a web page at a given URL",
            "Fetch and convert a webpage to plain text format",
        ],
        {
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL of the webpage to fetch (must start \
with http:// or https://)",
                }
            },
            "required": ["url"],
        },
        _tool_fetch_webpage,
    ),
}


def get_tool_response(name: str, args: str) -> str:
    """
    Retrieves a response for the
    """

    try:
        parsed_args = json.loads(args)
    except json.JSONDecodeError:
        logger.error("Tool call arguments were not valid JSON: %s", args)
        raise ValueError("Model did not return valid JSON for the tool call.")

    try:
        logger.debug("Calling tool '%s' with parsed args: %s", name, parsed_args)
        res = TOOLS[name].callback(parsed_args)
        logger.debug("Tool '%s' returned result type %s", name, type(res).__name__)
        return json.dumps(res)
    except (KeyError, ValueError) as e:
        logger.exception("Tool '%s' failed: %s", name, e)
        raise ToolError("The tool did not run successfully: ", e)


def generate_random_tool_selection(tools: list[str]) -> list[dict]:
    """
    From a given list of wanted tool keys, generates a selection of tools with
    variations in their names and descriptions, and returns them in the format
    expected for OpenAI API tool definitions.
    """

    selection = [
        json.loads(TOOLS[tool].generate_variation()) for tool in tools if tool in TOOLS
    ]
    logger.debug("Generated %d tool variations for tools: %s", len(selection), tools)
    return selection

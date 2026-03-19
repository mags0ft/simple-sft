"""
Defines types for use across simple-sft.
"""

from typing import Any, Optional, TypedDict


class BaseMessageType(TypedDict):
    role: str
    content: str


class ToolType(TypedDict):
    name: str
    description: str
    parameters: dict[str, Any]


class TopLevelToolType(TypedDict):
    type: str
    function: ToolType


class ToolCallFunctionType(TypedDict):
    name: str
    arguments: str


class ToolCallType(TypedDict):
    id: str
    type: str
    function: ToolCallFunctionType


class MessagesType(BaseMessageType, total=False):
    tool_calls: Optional[list[ToolCallType]]
    thinking: Optional[str]
    tool_call_id: Optional[str]


class ConversationType(TypedDict):
    messages: list[MessagesType]
    id: str
    category: str
    tools: list[TopLevelToolType]
    specials: str

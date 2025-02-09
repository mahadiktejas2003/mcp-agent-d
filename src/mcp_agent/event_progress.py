"""Module for converting log events to progress events."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Union
from mcp_agent.workflows.llm.llm_constants import FINAL_RESPONSE_LOG_MESSAGE


class ProgressAction(str, Enum):
    """Progress actions available in the system."""

    STARTING = "Starting"
    INITIALIZED = "Initialized"
    CHATTING = "Chatting"
    CALLING_TOOL = "Calling Tool"
    FINISHED = "Finished"
    SHUTDOWN = "Shutdown"
    AGGREGATOR_INITIALIZED = "Running"


class ProgressLogMessage(str, Enum):
    LOG_AGGREGATOR_INITIALIZED = (
        "MCP Aggregator initialized "  # TODO -- complete, and map LOG->Action
    )


@dataclass
class ProgressEvent:
    """Represents a progress event converted from a log event."""

    action: ProgressAction
    target: str
    details: Optional[str] = None
    agent_name: Optional[str] = None

    def __str__(self) -> str:
        """Format the progress event for display."""
        base = f"{self.action.ljust(11)}. {self.target}"
        if self.details:
            base += f" - {self.details}"
        if self.agent_name:
            base = f"[{self.agent_name}] {base}"
        return base


def extract_from_aggregator(message: str) -> Optional[str]:
    """Extract tool information from MCPServerAggregator messages."""
    if "Requesting tool call" in message:
        try:
            tool_name = message.split("'")[1]
            return tool_name
        except IndexError:
            pass
    return None


def convert_log_event(event: Dict[str, Any]) -> Optional[ProgressEvent]:
    """Convert a log event to a progress event if applicable."""
    namespace = event.get("namespace", "")
    message = event.get("message", "")

    # Extract agent name from namespace if present
    agent_name = "default"

    # Known class namespaces that may have agent names
    CLASS_NAMESPACES = [
        "mcp_agent.agents.agent",  # TODO: Use Agent.__module__
        ("mcp_agent.workflows.llm", "augmented_llm"),  # Matches augmented_llm_* classes
        "mcp_agent.mcp.mcp_aggregator",  # TODO: Use Finder.__module__
    ]

    # Check if namespace starts with any of our class prefixes and has an additional part
    for class_ns in CLASS_NAMESPACES:
        if isinstance(class_ns, tuple):
            # Special case for augmented_llm_* classes
            base_ns, class_prefix = class_ns
            parts = namespace[len(base_ns) + 1 :].split(".")  # +1 for the dot
            if len(parts) >= 2 and parts[0].startswith(class_prefix):
                agent_name = parts[1]
                break
        elif namespace.startswith(class_ns + "."):
            # Regular case - agent name is after the class namespace
            agent_name = namespace.split(".")[-1]
            break

    # Handle MCP connection events
    if "mcp_connection_manager" in namespace:
        # Extract MCP name from message prefix if present
        mcp_name = None
        if ": " in message:
            mcp_name = message.split(": ")[0]
            message = message.split(": ")[1]

        # Handle lifecycle events
        if "_lifecycle_task is exiting" in message:
            if mcp_name:
                return ProgressEvent(
                    ProgressAction.SHUTDOWN, f"MCP '{mcp_name}'", agent_name=agent_name
                )
        elif "Session initialized" in message:
            return ProgressEvent(
                ProgressAction.INITIALIZED, f"MCP '{mcp_name}'", agent_name=agent_name
            )
        elif "Initializing server session" in message:
            return ProgressEvent(
                ProgressAction.STARTING, f"MCP '{mcp_name}'", agent_name=agent_name
            )

    # Handle MCPServerAggregator tool calls
    if "mcp_aggregator" in namespace:
        tool_name = extract_from_aggregator(message)
        target = f"{agent_name} ({tool_name})" if agent_name else tool_name
        if tool_name:
            return ProgressEvent(
                ProgressAction.CALLING_TOOL, target, agent_name=agent_name
            )

        if ProgressLogMessage.LOG_AGGREGATOR_INITIALIZED in message:
            return ProgressEvent(
                ProgressAction.AGGREGATOR_INITIALIZED,
                "mcp-agent",
            )

    # Handle LLM events
    if "augmented_llm" in namespace:
        # Handle non-dict data safely
        data = event.get("data", {})
        if not isinstance(data, dict):
            data = {}  # If data isn't a dict, treat as empty

        model = data.get("model", "")
        chat_turn = data.get("chat_turn")

        # If not found, try nested data
        nested_data = data.get("data", {})
        if isinstance(nested_data, dict):
            if not model:
                model = nested_data.get("model", "")
            if chat_turn is None:
                chat_turn = nested_data.get("chat_turn")

        target = f"{agent_name} ({model})" if agent_name else model

        if "Calling " in message:
            if chat_turn is not None:
                return ProgressEvent(
                    ProgressAction.CHATTING,
                    target,
                    f"Turn {chat_turn}",
                    agent_name=agent_name,
                )
            return ProgressEvent(ProgressAction.CHATTING, target, agent_name=agent_name)

        elif FINAL_RESPONSE_LOG_MESSAGE in message:
            return ProgressEvent(ProgressAction.FINISHED, target, agent_name=agent_name)

    return None

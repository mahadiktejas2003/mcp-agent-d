"""Module for converting log events to progress events."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional
from mcp_agent.workflows.llm.llm_constants import FINAL_RESPONSE_LOG_MESSAGE


class ProgressAction(str, Enum):
    """Progress actions available in the system."""

    STARTING = "Starting"
    INITIALIZED = "Initialized"
    CHATTING = "Chatting"
    CALLING_TOOL = "Calling Tool"
    FINISHED = "Finished"
    SHUTDOWN = "Shutdown"


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
    agent_name = None
    parts = namespace.split(".")
    if len(parts) > 3:
        if parts[0:3] == ["mcp_agent", "agents", "agent"]:
            agent_name = parts[3]
        elif (
            len(parts) > 4
            and parts[0:3] == ["mcp_agent", "workflows", "llm"]
            and parts[3].startswith("augmented_llm")
        ):
            agent_name = parts[4]

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
        if tool_name:
            return ProgressEvent(
                ProgressAction.CALLING_TOOL, tool_name, agent_name=agent_name
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

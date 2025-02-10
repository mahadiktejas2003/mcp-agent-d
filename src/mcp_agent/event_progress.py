"""Module for converting log events to progress events."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Union


class ProgressAction(str, Enum):
    """Progress actions available in the system."""

    STARTING = "Starting"
    INITIALIZED = "Initialized"
    CHATTING = "Chatting"
    CALLING_TOOL = "Calling Tool"
    FINISHED = "Finished"
    SHUTDOWN = "Shutdown"
    AGGREGATOR_INITIALIZED = "Running"


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


def convert_log_event(event: Dict[str, Any]) -> Optional[ProgressEvent]:
    """Convert a log event to a progress event if applicable."""
    # Return early if event is not a dictionary
    if not isinstance(event, dict):
        return None

    # Get data field, defaulting to empty dict
    data = event.get("data", {})
    if not isinstance(data, dict):
        return None

    # Handle both nested and flat data structures
    progress_data = data.get("data", data)
    progress_action = progress_data.get("progress_action")
    if not progress_action:
        return None

    # Build target string based on the event type
    namespace = event.get("namespace", "")
    if "mcp_connection_manager" in namespace:
        target = f"MCP '{data.get('mcp_name')}'"
    elif "mcp_aggregator" in namespace:
        server_name = data.get("server_name", "")
        tool_name = data.get("tool_name", "")
        target = f"{server_name} ({tool_name})" if server_name else tool_name
    elif "augmented_llm" in namespace:
        model = data.get("model", "")
        agent_name = data.get("agent_name")
        target = f"{agent_name} ({model})" if agent_name else model
        # Add chat turn if present
        chat_turn = data.get("data", {}).get("chat_turn")
        if chat_turn is not None:
            return ProgressEvent(
                ProgressAction(progress_action),
                target,
                f"Turn {chat_turn}",
                agent_name=data.get("agent_name"),
            )
    else:
        target = data.get("target", "unknown")

    return ProgressEvent(
        ProgressAction(progress_action), target, agent_name=data.get("agent_name")
    )

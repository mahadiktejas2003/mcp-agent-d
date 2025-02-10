"""Module for converting log events to progress events."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from mcp_agent.logging.events import Event


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


def convert_log_event(event: Event) -> Optional[ProgressEvent]:
    """Convert a log event to a progress event if applicable."""

    # Check to see if there is any additional data
    if not event.data:
        return None

    event_data = event.data.get("data")
    if not isinstance(event_data, dict):
        return None

    progress_action = event_data.get("progress_action")
    if not progress_action:
        return None

    # Build target string based on the event type
    namespace = event.namespace
    if "mcp_connection_manager" in namespace:
        target = f"MCP '{event_data.get('mcp_name')}'"
    elif "mcp_aggregator" in namespace:
        server_name = event_data.get("server_name", "")
        tool_name = event_data.get("tool_name", "")
        target = f"{server_name} ({tool_name})" if server_name else tool_name
    elif "augmented_llm" in namespace:
        model = event_data.get("model", "")
        agent_name = event_data.get("agent_name")
        target = f"{agent_name} ({model})" if agent_name else model
        # Add chat turn if present
        chat_turn = event_data.get("chat_turn")
        if chat_turn is not None:
            return ProgressEvent(
                ProgressAction(progress_action),
                target,
                f"Turn {chat_turn}",
                agent_name=event_data.get("agent_name"),
            )
    else:
        target = event_data.get("target", "unknown")

    return ProgressEvent(
        ProgressAction(progress_action),
        target,
        agent_name=event_data.get("agent_name"),
    )

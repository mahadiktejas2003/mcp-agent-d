"""Module for converting log events to progress events."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


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
    
    def __str__(self) -> str:
        """Format the progress event for display."""
        base = f"{self.action.ljust(11)}. {self.target}"
        if self.details:
            base += f" - {self.details}"
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
                return ProgressEvent(ProgressAction.SHUTDOWN, f"MCP '{mcp_name}'")
        elif "Session initialized" in message:
            return ProgressEvent(ProgressAction.INITIALIZED, f"MCP '{mcp_name}'")
        elif "Initializing server session" in message:
            return ProgressEvent(ProgressAction.STARTING, f"MCP '{mcp_name}'")
    
    # Handle MCPServerAggregator tool calls
    if "mcp_aggregator" in namespace:
        tool_name = extract_from_aggregator(message)
        if tool_name:
            return ProgressEvent(ProgressAction.CALLING_TOOL, tool_name)
    
    # Handle LLM events
    if "augmented_llm" in namespace:

        llm_type = "llm_openai" if "openai" in namespace else "llm_anthropic"
        
        if "Calling OpenAI ChatCompletion" in message or "Calling claude" in message:
            try:
                iteration = message.split("Iteration")[1].split(":")[0].strip()
                turn = int(iteration)
                return ProgressEvent(ProgressAction.CHATTING, llm_type, f"Iteration {turn}")
            except (ValueError, IndexError):
                return ProgressEvent(ProgressAction.CHATTING, llm_type)

        elif "Finished processing" in message or "Completed request" in message:
            return ProgressEvent(ProgressAction.FINISHED, llm_type, message)
    
    return None

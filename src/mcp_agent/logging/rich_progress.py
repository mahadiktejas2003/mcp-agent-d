"""Rich-based progress display for MCP Agent."""

from typing import Optional
import sys
from rich.console import Console
from rich.text import Text
from mcp_agent.event_progress import ProgressEvent, ProgressAction


class RichProgressDisplay:
    """Rich-based display for progress events."""

    def __init__(self, console: Optional[Console] = None):
        """Initialize the progress display."""
        self.console = console or Console()

    def start(self):
        """start"""

    def stop(self):
        """stop"""

    def _get_action_style(self, action: ProgressAction) -> str:
        """Map actions to appropriate styles."""
        return {
            ProgressAction.STARTING: "black on yellow",
            ProgressAction.INITIALIZED: "black on green",
            ProgressAction.CHATTING: "white on blue",
            ProgressAction.CALLING_TOOL: "white on magenta",
            ProgressAction.FINISHED: "black on green",
            ProgressAction.SHUTDOWN: "black on red",
        }.get(action, "white")

    def _format_event(self, event: ProgressEvent) -> Text:
        """Format a single event as rich Text."""
        text = Text()
        text.append(
            f" {event.action.value:<15}", style=self._get_action_style(event.action)
        )
        text.append(" ")
        text.append(f"{event.target}", style="white bold")
        if event.details:
            text.append(f" ({event.details})", style="blue")
        return text

    def update(self, event: ProgressEvent) -> None:
        """Update with a new progress event."""
        self.console.print(self._format_event(event))

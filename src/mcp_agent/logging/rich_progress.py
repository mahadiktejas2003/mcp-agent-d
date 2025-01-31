"""Rich-based progress display for MCP Agent."""

from typing import Optional, List
from rich.console import Console
from rich.style import Style
from rich.text import Text
from rich.padding import Padding
from rich.box import SQUARE
from rich.panel import Panel
from rich.status import Status

from mcp_agent.event_progress import ProgressEvent, ProgressAction


class RichProgressDisplay:
    """Rich-based display for progress events."""

    ACTION_WIDTH = 15  # Width for the action field

    def __init__(self, console: Optional[Console] = None):
        """Initialize the progress display."""
        self.console = console or Console()
        self.events: List[Text] = []
        self.status: Optional[Status] = None

    def _render(self) -> Panel:
        """Render the current state as a panel."""
        content = Text("\n").join(self.events)
        return Panel(
            content,
            title="Agent Progress",
            title_align="left",
            box=SQUARE,
            padding=(0, 1),
            style="blue",
            expand=True,
        )

    def start(self):
        """Start the display."""
        self.status = self.console.status("")
        self.status.start()

    def stop(self):
        """Stop the display."""
        if self.status:
            # Render final state
            self.console.print(self._render())
            self.status.stop()

    def _get_action_style(self, action: ProgressAction) -> str:
        """Get the appropriate style for an action."""
        return {
            ProgressAction.STARTING: "yellow",
            ProgressAction.INITIALIZED: "green",
            ProgressAction.CHATTING: "blue",
            ProgressAction.CALLING_TOOL: "magenta",
            ProgressAction.FINISHED: "cyan bold",  # Made more prominent
            ProgressAction.SHUTDOWN: "red",
        }.get(action, "white")

    def update(self, event: ProgressEvent):
        """Display a new progress event."""
        text = Text()

        # Action field (padded to ACTION_WIDTH)
        text.append(
            event.action.value.ljust(self.ACTION_WIDTH),
            style=self._get_action_style(event.action),
        )

        # Target
        text.append(f"{event.target}", style="bold")

        # Details (if any)
        if event.details:
            text.append(f" ({event.details})", style="dim")

        self.events.append(text)
        if self.status:
            self.status.update(self._render())

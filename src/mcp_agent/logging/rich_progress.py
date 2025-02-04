"""Rich-based progress display for MCP Agent."""

from typing import Optional
from rich.console import Console
from rich.text import Text
from mcp_agent.event_progress import ProgressEvent, ProgressAction
from rich.progress import Progress, TextColumn, TimeElapsedColumn


class RichProgressDisplay:
    """Rich-based display for progress events."""

    def __init__(self, console: Optional[Console] = None):
        """Initialize the progress display."""
        self.console = console or Console()

    def start(self):
        """start"""
        with Progress(
            TimeElapsedColumn(),
            *Progress.get_default_columns(),
            TextColumn(text_format="1234"),
            TextColumn(text_format="5678"),
            console=self.console,
            transient=False,
        ) as self._progress:
            self._task1 = self._progress.add_task("...............", total=None)
        self._progress.start()

    def stop(self):
        """stop"""
        self._progress.stop()

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
            f" {event.action.value:<14}", style=self._get_action_style(event.action)
        )
        text.append(" ")
        text.append(f"{event.target}", style="white bold")
        if event.details:
            text.append(f" ({event.details})", style="blue")
        return text

    def update(self, event: ProgressEvent) -> None:
        self._progress.update(
            self._task1,
            description=f"[{self._get_action_style(event.action)}]{event.action.value:<15}",
        )

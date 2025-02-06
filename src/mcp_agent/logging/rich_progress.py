"""Rich-based progress display for MCP Agent."""

from typing import Optional
from rich.console import Console
from rich.text import Text
from mcp_agent.event_progress import ProgressEvent, ProgressAction
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn


class RichProgressDisplay:
    """Rich-based display for progress events."""

    def __init__(self, console: Optional[Console] = None):
        """Initialize the progress display."""
        self.console = console or Console()
        self._taskmap = {}
        self._progress = None

    def start(self):
        """start"""
        with Progress(
            TimeElapsedColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=15),
            TextColumn(text_format="{task.fields[target]:<18}", style="Blue"),
            TextColumn(text_format="{task.fields[details]}"),
            console=self.console,
            transient=False,
        ) as self._progress:
            task_id = self._progress.add_task(
                "mcp-agent......", total=None, target="", details=""
            )
            self._taskmap["default"] = task_id

        self._progress.start()

    def stop(self):
        """stop"""
        default_task_id = self._taskmap["default"]
        self._progress.update(default_task_id, total=100, completed=100)
        self._progress.stop_task(default_task_id)

        self._progress.stop()

    def _get_action_style(self, action: ProgressAction) -> str:
        """Map actions to appropriate styles."""
        return {
            ProgressAction.STARTING: "black on yellow",
            ProgressAction.INITIALIZED: "black on green",
            ProgressAction.CHATTING: "white on dark_blue",
            ProgressAction.CALLING_TOOL: "white on dark_magenta",
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
        task_name = event.agent_name or "default"
        if task_name not in self._taskmap:
            task_id = self._progress.add_task(
                "",
                total=None,
                target=f"{event.target}",
                details=f"{event.agent_name}",
            )
            self._taskmap[event.agent_name] = task_id
        else:
            task_id = self._taskmap[task_name]

        if event.action == ProgressAction.FINISHED:
            self._progress.update(task_id, total=100, completed=100)
            self._progress.stop_task(task_id)

        self._progress.update(
            task_id,
            description=f"[{self._get_action_style(event.action)}]{event.action.value:<15}",
            target=event.target,
            details=event.details if event.details else "",
        )

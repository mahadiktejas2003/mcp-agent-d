#!/usr/bin/env python3
"""MCP Event Viewer"""

import json
import sys
import tty
import termios
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text


def get_key() -> str:
    """Get a single keypress."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


class EventDisplay:
    """Display MCP events from a log file."""

    def __init__(self, events: list):
        self.events = events
        self.total = len(events)
        self.current = 0
        self.current_llm: Optional[str] = None
        self.current_iteration: Optional[int] = None
        self.tool_calls = 0

    def next(self) -> None:
        """Move to next event."""
        if self.current < self.total:
            self._process_event(self.events[self.current])
            self.current += 1

    def prev(self) -> None:
        """Move to previous event."""
        if self.current > 0:
            # Reset state and replay up to previous event
            self._reset_state()
            for i in range(self.current - 1):
                self._process_event(self.events[i])
            self.current -= 1

    def _reset_state(self) -> None:
        """Reset state for replay."""
        self.current_llm = None
        self.current_iteration = None
        self.tool_calls = 0

    def _process_event(self, event: dict) -> None:
        """Update state based on event."""
        namespace = event.get("namespace", "")
        message = event.get("message", "")

        # Track LLM switches
        if "augmented_llm_openai" in namespace:
            self.current_llm = "OpenAI"
        elif "augmented_llm_anthropic" in namespace:
            self.current_llm = "Anthropic"

        # Track iterations
        if "Iteration" in message:
            try:
                self.current_iteration = int(
                    message.split("Iteration")[1].split(":")[0]
                )
            except (ValueError, IndexError):
                pass

        # Track tool calls
        if "Tool call" in message or "Calling tool" in message:
            self.tool_calls += 1

    def render(self) -> Panel:
        """Render current event state."""
        text = Text()

        # Show LLM
        text.append("Current LLM: ", style="bold")
        text.append(f"{self.current_llm or 'None'}\n", style="green")

        # Show Iteration
        text.append("Iteration: ", style="bold")
        text.append(f"{self.current_iteration or 'None'}\n", style="blue")

        # Show Stats
        text.append("\nStats:\n", style="bold")
        text.append(f"Event: {self.current}/{self.total}\n", style="cyan")
        text.append(f"Tool Calls: {self.tool_calls}\n", style="magenta")

        # Show Controls
        text.append("\nh/l to move • q to quit", style="dim")

        return Panel(text, title="MCP Event Viewer")


def load_events(path: Path) -> list:
    """Load events from JSONL file."""
    events = []
    with open(path) as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    return events


def main(log_file: str):
    """View MCP Agent events from a log file."""
    events = load_events(Path(log_file))
    display = EventDisplay(events)
    console = Console()

    with Live(
        display.render(),
        console=console,
        screen=True,
        transient=True,
        auto_refresh=False,  # Only refresh on demand
    ) as live:
        while True:
            key = get_key()

            if key in {"l", "L"}:  # Next
                display.next()
            elif key in {"h", "H"}:  # Previous
                display.prev()
            elif key in {"q", "Q"}:  # Quit
                break

            live.update(display.render())
            live.refresh()


if __name__ == "__main__":
    typer.run(main)

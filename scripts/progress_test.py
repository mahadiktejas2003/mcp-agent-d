#!/usr/bin/env python3
"""Simple progress display test program."""

import sys
import tty
import termios

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


class SimpleDisplay:
    def __init__(self, total: int):
        self.current = 0
        self.total = total

    def next(self) -> None:
        """Move forward (l or right)"""
        if self.current < self.total:
            self.current += 1

    def prev(self) -> None:
        """Move backward (h or left)"""
        if self.current > 0:
            self.current -= 1

    def render(self) -> Panel:
        """Render current progress state"""
        text = Text()
        text.append(f"Progress: {self.current}/{self.total}\n", style="bold blue")
        text.append("\nh/l or ←/→ to move • q to quit", style="dim")
        return Panel(text, title="Test Display")


def main():
    display = SimpleDisplay(total=75)
    console = Console()

    with Live(
        display.render(),
        console=console,
        screen=True,  # Use alternate screen
        transient=True,  # Remove display on exit
        auto_refresh=False,
        # Only refresh when explicitly called
    ) as live:
        while True:
            key = get_key()
            print(f"{key}")
            if key in {"l", "L"}:  # Next
                print(f"{key} *******DOWN")
                display.next()
                live.update(display.render())
            elif key in {"h", "H"}:  # Previous
                display.prev()
                live.update(display.render())
            elif key in {"q", "Q"}:  # Quit
                break
            live.refresh()


if __name__ == "__main__":
    main()

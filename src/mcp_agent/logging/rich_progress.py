"""Rich-based progress display for MCP Agent."""

from dataclasses import dataclass
from typing import Dict, Optional
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.style import Style

from mcp_agent.event_progress import ProgressEvent, ProgressAction

class RichProgressDisplay:
    """Rich-based display for progress events."""
    
    def __init__(self, console: Optional[Console] = None):
        """Initialize the progress display.
        
        Args:
            console: Optional Rich console to use for display
        """
        self.console = console or Console()
        # Track active events by target
        self.active_events: Dict[str, ProgressEvent] = {}
        self.live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=4,
            transient=True
        )
        
    def start(self):
        """Start the live display."""
        self.live.start()
        
    def stop(self):
        """Stop the live display."""
        self.live.stop()
        
    def _get_action_style(self, action: ProgressAction) -> str:
        """Get the appropriate style for an action."""
        return {
            ProgressAction.STARTING: "yellow",
            ProgressAction.INITIALIZED: "green",
            ProgressAction.CHATTING: "blue",
            ProgressAction.CALLING_TOOL: "magenta",
            ProgressAction.FINISHED: "green",
            ProgressAction.SHUTDOWN: "red"
        }.get(action, "white")
        
    def _render(self) -> Panel:
        """Render the current state of all activities."""
        if not self.active_events:
            return Panel(
                Text("No active processes", style="dim italic"),
                title="Agent Progress Monitor",
                border_style="blue"
            )
            
        table = Table.grid(padding=(0, 2))
        table.add_column("Action", justify="right")
        table.add_column("Target", style="bold")
        table.add_column("Details", style="dim")
        
        # Sort by target for consistent display
        for target in sorted(self.active_events.keys()):
            event = self.active_events[target]
            action_style = self._get_action_style(event.action)
            
            table.add_row(
                event.target,
                Text(event.action, style=action_style),
                event.details or ""
            )
            
        return Panel(
            table,
            title="MCP Progress Monitor",
            border_style="blue"
        )
        
    def update(self, event: ProgressEvent):
        """Update the display with a new progress event."""
        # Update or add the event for this target
        self.active_events[event.target] = event
        
        # Remove finished/shutdown targets after a brief delay
        if event.action in [ProgressAction.FINISHED, ProgressAction.SHUTDOWN]:
            # In a real implementation, you might want to use asyncio.create_task 
            # to remove this after a delay, but for now we'll keep it visible
            pass
            
        self.live.update(self._render())

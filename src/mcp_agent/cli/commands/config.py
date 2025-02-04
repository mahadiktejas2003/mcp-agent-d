import typer
from rich import print

app = typer.Typer()


@app.command()
def show():
    """Show the configuration."""
    print("NotImplemented")

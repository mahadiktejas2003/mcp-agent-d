"""
FastMCP File Operations Server

Provides file system and URL operations with automatic root directory management.
"""

import os
import aiohttp
from mcp.server.fastmcp import FastMCP, Context
from typing import Optional, Annotated
from pathlib import Path
from urllib.parse import urlparse
from pydantic import BaseModel, Field, AnyHttpUrl

# Create server
mcp = FastMCP("File Operations Server")


async def check_and_change_root(ctx: Context) -> Optional[str]:
    """Helper function to check and change to root directory if available"""
    roots = await ctx.session.list_roots()
    if roots:
        first_root = roots[0]
        os.chdir(first_root)
        return first_root
    return None


@mcp.tool()
async def read_file(
    ctx: Context,
    filepath: Annotated[str, Field(description="Path to the file to read. Can be relative to current directory or absolute.")]
) -> str:
    """Read the content of a file from the filesystem."""
    await check_and_change_root(ctx)
    
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")
        
    with open(path, 'r') as f:
        return f.read()


@mcp.tool()
async def write_file(
    ctx: Context,
    filepath: Annotated[str, Field(description="Path where the file should be written. Can be relative to current directory or absolute.")],
    content: Annotated[str, Field(description="The text content to write to the file")]
) -> str:
    """Write content to a file on the filesystem."""
    await check_and_change_root(ctx)
    
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
        
    with open(path, 'w') as f:
        f.write(content)
    
    return f"Successfully wrote to {path}"


@mcp.tool()
async def fetch_url(
    url: Annotated[AnyHttpUrl, Field(description="The URL to fetch content from")]
) -> str:
    """Fetch content from a URL."""
    async with aiohttp.ClientSession() as session:
        async with session.get(str(url)) as response:
            response.raise_for_status()
            return await response.text()


@mcp.tool()
async def save_url_to_file(
    ctx: Context,
    url: Annotated[AnyHttpUrl, Field(description="The URL to fetch content from")],
    filepath: Annotated[str, Field(description="Path where the file should be written. Can be relative to current directory or absolute.")]
) -> str:
    """Fetch content from a URL and save it directly to a file."""
    await check_and_change_root(ctx)
    
    # Show progress using chunks
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    async with aiohttp.ClientSession() as session:
        async with session.get(str(url)) as response:
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            
            with open(path, 'wb') as f:
                bytes_read = 0
                async for chunk in response.content.iter_chunked(8192):
                    f.write(chunk)
                    bytes_read += len(chunk)
                    if total_size:
                        await ctx.report_progress(bytes_read, total_size)
            
            return f"Successfully saved {url} to {path}"


if __name__ == "__main__":
    mcp.run()
"""
Custom implementation of stdio_client that handles stderr through rich console.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, TextIO

import platform
import shutil
import subprocess

import anyio
import anyio.lowlevel
from anyio.streams.text import TextReceiveStream
from mcp.client.stdio import StdioServerParameters, get_default_environment
import mcp.types as types
from mcp_agent.logging.logger import get_logger

logger = get_logger(__name__)


def get_executable_command(command: str) -> str:
    """
    Get the correct executable command normalized for the current platform.

    Args:
        command: Base command (e.g., 'uvx', 'npx')
        args: Command arguments

    Returns:
        List[str]: Platform-appropriate command
    """

    try:
        if platform.system() != "Windows":
            return command
        else:
            # For Windows, we need more sophisticated path resolution
            # First check if command exists in PATH as-is
            command_path = shutil.which(command)
            if command_path:
                logger.debug(f"Found command '{command} in PATH: {command_path}")
                return command_path

            # Check for Windows-specific extensions
            for ext in [".cmd", ".bat", ".exe", ".ps1"]:
                ext_version = f"{command}{ext}"
                ext_path = shutil.which(ext_version)
                if ext_path:
                    logger.debug(f"Found command {command} with extension: {ext_path}")
                    return ext_path

            # For regular commands or if we couldn't find special versions
            logger.debug(
                f"Warning: Couldn't shutil.which({command}). Using original '{command}'"
            )
            return command
    except Exception as exc:
        logger.warning(
            f"Error in get_executable_command, defaulting to original '{command}'. Error: {exc}"
        )
        return command


async def create_platform_compatible_process(
    command: str,
    args: List[str],
    env: Dict[str, str] | None = None,
    errlog: int | TextIO = subprocess.PIPE,
    cwd: Path | str | None = None,
):
    """
    Creates a subprocess in a platform-compatible way.
    Returns a process handle.
    """

    process = None

    if platform.system() == "Windows":
        try:
            process = await anyio.open_process(
                [command, *args],
                env=env,
                # Ensure we don't create console windows for each process
                creationflags=subprocess.CREATE_NO_WINDOW
                if hasattr(subprocess, "CREATE_NO_WINDOW")
                else 0,
                stderr=errlog,
                cwd=cwd,
            )

            return process
        except Exception as e:
            logger.warning(
                f"Error creating subprocess using create_subprocess_exec for '{command}': {e}",
                exc_info=True,
            )
            # Don't raise, let's try to create the process using the default method
            process = None

    # Default method for creating the process
    process = await anyio.open_process(
        [command, *args], env=env, stderr=errlog, cwd=cwd
    )

    return process


@asynccontextmanager
async def stdio_client_with_rich_stderr(
    server: StdioServerParameters, errlog: int | TextIO = subprocess.PIPE
):
    """
    Modified version of stdio_client that captures stderr and routes it through our rich console.
    Follows the original pattern closely for reliability.

    Args:
        server: The server parameters for the stdio connection
    """
    read_stream_writer, read_stream = anyio.create_memory_object_stream(0)
    write_stream, write_stream_reader = anyio.create_memory_object_stream(0)

    command = get_executable_command(server.command)

    # Open process with stderr piped for capture
    process = await create_platform_compatible_process(
        command=command,
        args=server.args,
        env=(
            {**get_default_environment(), **server.env}
            if server.env is not None
            else get_default_environment()
        ),
        errlog=errlog,
    )

    if process.pid:
        logger.debug(f"Started process '{command}' with PID: {process.pid}")

    if process.returncode is not None:
        logger.debug(f"return code (early){process.returncode}")
        raise RuntimeError(
            f"Process terminated immediately with code {process.returncode}"
        )

    async def stdout_reader():
        assert process.stdout, "Opened process is missing stdout"
        try:
            async with read_stream_writer:
                buffer = ""
                async for chunk in TextReceiveStream(
                    process.stdout,
                    encoding=server.encoding,
                    errors=server.encoding_error_handler,
                ):
                    lines = (buffer + chunk).split("\n")
                    buffer = lines.pop()

                    for line in lines:
                        if not line:
                            continue
                        try:
                            message = types.JSONRPCMessage.model_validate_json(line)
                        except Exception as exc:
                            await read_stream_writer.send(exc)
                            continue

                        await read_stream_writer.send(message)
        except anyio.ClosedResourceError:
            await anyio.lowlevel.checkpoint()
        except Exception as e:
            logger.error(f"Error in stdout_reader: {e}")

    async def stderr_reader():
        assert process.stderr, "Opened process is missing stderr"
        try:
            async for chunk in TextReceiveStream(
                process.stderr,
                encoding=server.encoding,
                errors=server.encoding_error_handler,
            ):
                if chunk.strip():
                    # Let the logging system handle the formatting consistently
                    logger.event("info", "mcpserver.stderr", chunk.rstrip(), None, {})
        except anyio.ClosedResourceError:
            await anyio.lowlevel.checkpoint()
        except Exception as e:
            logger.error(f"Error in stderr_reader: {e}")

    async def stdin_writer():
        assert process.stdin, "Opened process is missing stdin"
        try:
            async with write_stream_reader:
                async for message in write_stream_reader:
                    json = message.model_dump_json(by_alias=True, exclude_none=True)
                    data = (json + "\n").encode(
                        encoding=server.encoding, errors=server.encoding_error_handler
                    )

                    await process.stdin.send(data)
        except anyio.ClosedResourceError:
            await anyio.lowlevel.checkpoint()
        except Exception as e:
            logger.error(f"Error in stdin_writer: {e}")

    # Use context managers to handle cleanup automatically
    async with anyio.create_task_group() as tg, process:
        tg.start_soon(stdout_reader)
        tg.start_soon(stdin_writer)
        tg.start_soon(stderr_reader)
        yield read_stream, write_stream

"""
Custom implementation of stdio_client that handles stderr through rich console.
"""

from contextlib import asynccontextmanager, suppress
import asyncio
import subprocess
import anyio
import anyio.lowlevel
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from anyio.streams.text import TextReceiveStream
from mcp.client.stdio import StdioServerParameters, get_default_environment
import mcp.types as types
from mcp_agent.logging.logger import get_logger
from enum import Enum
from dataclasses import dataclass

logger = get_logger(__name__)

class StderrLogLevel(str, Enum):
    """Log levels that can be used for stderr output."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class StdioClientConfig:
    """Configuration for stdio client behavior."""
    stderr_log_level: StderrLogLevel = StderrLogLevel.WARNING  # Default to WARNING level
    
@asynccontextmanager
async def stdio_client_with_rich_stderr(
    server: StdioServerParameters,
    config: StdioClientConfig = StdioClientConfig()
):
    """
    Modified version of stdio_client that captures stderr and routes it through our rich console.
    
    Args:
        server: The server parameters for the stdio connection
        config: Configuration for the stdio client behavior
    """
    read_stream: MemoryObjectReceiveStream[types.JSONRPCMessage | Exception]
    read_stream_writer: MemoryObjectSendStream[types.JSONRPCMessage | Exception]

    write_stream: MemoryObjectSendStream[types.JSONRPCMessage]
    write_stream_reader: MemoryObjectReceiveStream[types.JSONRPCMessage]

    read_stream_writer, read_stream = anyio.create_memory_object_stream(0)
    write_stream, write_stream_reader = anyio.create_memory_object_stream(0)

    # Map log level to logger function - we'll keep this for potential filtering
    log_func = getattr(logger, config.stderr_log_level.value)

    process = await anyio.open_process(
        [server.command, *server.args],
        env=server.env if server.env is not None else get_default_environment(),
        stderr=subprocess.PIPE,
    )

    # Flag to signal tasks to shut down
    shutdown_event = anyio.Event()

    async def stdout_reader():
        assert process.stdout, "Opened process is missing stdout"

        try:
            async with read_stream_writer:
                buffer = ""
                while not shutdown_event.is_set():
                    try:
                        chunk = await process.stdout.receive(4096)
                        if not chunk:  # EOF
                            break
                        text = chunk.decode(
                            encoding=server.encoding,
                            errors=server.encoding_error_handler,
                        )
                        lines = (buffer + text).split("\n")
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
                    except (anyio.EndOfStream, anyio.ClosedResourceError):
                        break
        except Exception as e:
            logger.error(f"Error in stdout reader: {e}")
        finally:
            await anyio.lowlevel.checkpoint()

    async def stderr_reader():
        assert process.stderr, "Opened process is missing stderr"

        try:
            while not shutdown_event.is_set():
                try:
                    chunk = await process.stderr.receive(4096)
                    if not chunk:  # EOF
                        break
                    text = chunk.decode(
                        encoding=server.encoding,
                        errors=server.encoding_error_handler,
                    )
                    if text.strip():
                        # Route through logger instead of direct console print
                        # This will integrate with the progress display
                        logger.debug(f"[dim][Server stderr] {text.rstrip()}[/dim]")
                except (anyio.EndOfStream, anyio.ClosedResourceError):
                    break
        except Exception as e:
            logger.error(f"Error in stderr reader: {e}")
        finally:
            await anyio.lowlevel.checkpoint()

    async def stdin_writer():
        assert process.stdin, "Opened process is missing stdin"

        try:
            async with write_stream_reader:
                async for message in write_stream_reader:
                    if shutdown_event.is_set():
                        break
                    try:
                        json = message.model_dump_json(by_alias=True, exclude_none=True)
                        await process.stdin.send(
                            (json + "\n").encode(
                                encoding=server.encoding,
                                errors=server.encoding_error_handler,
                            )
                        )
                    except (anyio.EndOfStream, anyio.ClosedResourceError):
                        break
        except Exception as e:
            logger.error(f"Error in stdin writer: {e}")
        finally:
            await anyio.lowlevel.checkpoint()

    async def cleanup():
        """Clean up resources in a controlled manner"""
        shutdown_event.set()
        
        # Give a short time for tasks to notice shutdown event
        await anyio.sleep(0.1)
        
        # Close streams in order
        if process.stdin:
            with suppress(Exception):
                await process.stdin.aclose()
        
        if process.stdout:
            with suppress(Exception):
                await process.stdout.aclose()
        
        if process.stderr:
            with suppress(Exception):
                await process.stderr.aclose()
        
        # Terminate process if still running
        with suppress(Exception):
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    process.kill()  # Force kill if it doesn't terminate

    try:
        async with anyio.create_task_group() as tg:
            tg.start_soon(stdout_reader)
            tg.start_soon(stdin_writer)
            tg.start_soon(stderr_reader)
            yield read_stream, write_stream
    finally:
        await cleanup()
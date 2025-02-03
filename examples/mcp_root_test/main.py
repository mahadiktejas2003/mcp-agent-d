import asyncio
import os
from pathlib import Path
import time

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.mcp.mcp_agent_client_session import MCPAgentClientSession
from mcp_agent.mcp.mcp_connection_manager import MCPConnectionManager
from mcp_agent.logging.events import EventFilter
from mcp_agent.logging.transport import FileTransport, AsyncEventBus
from mcp_agent.logging.logger import LoggingConfig

app = MCPApp(name="mcp_root_test")


async def example_usage():
    async with app.run() as agent_app:
        logger = agent_app.logger
        context = agent_app.context

        logger.info("Current config:", data=context.config.model_dump())

        connection_manager = MCPConnectionManager(context.server_registry)
        await connection_manager.__aenter__()

        try:
            root_test = await connection_manager.get_server(
                server_name="root-test", client_session_factory=MCPAgentClientSession
            )
            logger.info("root-test")

            result = await root_test.session.list_tools()
            logger.info("root-test: Tools available:", data=result.model_dump())

            result = await root_test.session.call_tool("show_roots")
            logger.error("root-test: show_roots:" + str(result))
        finally:
            await connection_manager.disconnect_server(server_name="root-test")
            logger.info("root-test: Disconnected from server.")
            await connection_manager.__aexit__(None, None, None)


if __name__ == "__main__":
    start = time.time()
    asyncio.run(example_usage())
    end = time.time()
    t = end - start

    print(f"Total run time: {t:.2f}s")

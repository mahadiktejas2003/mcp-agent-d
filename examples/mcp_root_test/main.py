import asyncio
import time

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.mcp.mcp_connection_manager import MCPConnectionManager
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.logging.logger import LoggingConfig

app = MCPApp(name="mcp_root_test")


async def example_usage():
    async with app.run() as agent_app:
        logger = agent_app.logger
        context = agent_app.context

        logger.info("Current config:", data=context.config.model_dump())

        connection_manager = MCPConnectionManager(context.server_registry)
        await connection_manager.__aenter__()

        # root_test = await connection_manager.get_server(
        #     server_name="root-test", client_session_factory=MCPAgentClientSession
        # )
        # logger.info("root-test")
        # int_test = await connection_manager.get_server(
        #     server_name="interpreter", client_session_factory=MCPAgentClientSession
        # )
        # result = await root_test.session.list_tools()
        # logger.info("root-test: Tools available:", data=result.model_dump())

        # result = await root_test.session.call_tool("show_roots")

        interpreter_agent = Agent(
            name="analysis",
            instruction="""You have access to a python interpreter.""",
            server_names=["root_test", "interpreter"],
        )

        llm = await interpreter_agent.attach_llm(AnthropicAugmentedLLM)

        result = await llm.generate_str(
            "call the show_roots tool and tell me what the result was"
        )
        logger.info(result)

        # await llm.generate_str("Write a short python program to reverse a string")
        # result = await llm.generate_str("Use the program to reverse 'hello world!!!'")
        # logger.info(result)

        # result = await llm.generate_str(
        #     "Write a python program to describe the filesystem, run it and show the result."
        # )
        # # result = await llm.generate_str("Use the program to reverse 'hello world!!!'")
        # logger.info(result)

        # result = await llm.generate_str(
        #     "Write a python program to tell me the current working directory"
        # )
        # result = await llm.generate_str("Use the program to reverse 'hello world!!!'")
        logger.info(result)

        result = await llm.generate_str(
            "There is a CSV file in the  /mnt/data/ directory. Use the Python Interpreter to to analyze the file. "
            + "Produce a detailed description of the data, and any patterns it contains. "
        )
        logger.info(result)

    await LoggingConfig.shutdown()


if __name__ == "__main__":
    start = time.time()
    asyncio.run(example_usage())
    end = time.time()
    t = end - start

    print(f"Total run time: {t:.2f}s")

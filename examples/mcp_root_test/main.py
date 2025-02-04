import asyncio
import time

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.mcp.mcp_connection_manager import MCPConnectionManager
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.logging.logger import LoggingConfig
from rich import print

app = MCPApp(name="mcp_root_test")


async def example_usage():
    async with app.run() as agent_app:
        logger = agent_app.logger
        context = agent_app.context

        logger.info("Current config:", data=context.config.model_dump())

        async with MCPConnectionManager(context.server_registry) as connection_manager:
            interpreter_agent = Agent(
                name="analysis",
                instruction="""You have access to a python interpreter.""",
                server_names=["root_test", "interpreter"],
            )

            try:
                llm = await interpreter_agent.attach_llm(AnthropicAugmentedLLM)

                result = await llm.generate_str(
                    "call the show_roots tool and tell me what the result was"
                )
                #               logger.info(result)

                # (claude does not need this signpost - this is where 'available files' pattern would be useful)
                result = await llm.generate_str(
                    "There is a file named '2024-10-26-test-data.csv' in the current directory. Use the Python Interpreter to to analyze the file. "
                    #                    "There is a CSV file in the current directory. Use the Python Interpreter to to analyze the file. "
                    + "Produce a detailed description of the data, and any patterns it contains. "
                )
                #                logger.info(result)

                result = await llm.generate_str(
                    "Use MatPlotLib to produce some insightful visualisations - save them as .png files "
                )
                logger.info(result)

            finally:
                # Clean up the agent
                await interpreter_agent.close()

    # Ensure logging is properly shutdown
    await LoggingConfig.shutdown()


if __name__ == "__main__":
    start = time.time()
    try:
        asyncio.run(example_usage())
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt, shutting down gracefully...")
    except Exception as e:
        print(f"Error during execution: {e}")
        raise
    finally:
        end = time.time()
        t = end - start
        print(f"Total run time: {t:.2f}s")

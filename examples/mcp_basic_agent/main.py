import asyncio
import os
from pathlib import Path
import time

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.logging.events import EventFilter
from mcp_agent.logging.transport import FileTransport, AsyncEventBus
from mcp_agent.logging.logger import LoggingConfig

# # Create logs directory if it doesn't exist
# LOGS_DIR = Path("logs")
# LOGS_DIR.mkdir(exist_ok=True)

# # Create a timestamp-based log file
# timestamp = time.strftime("%Y%m%d_%H%M%S")
# log_file = LOGS_DIR / f"mcp_basic_agent_{timestamp}.jsonl"

# # Create file transport with no filtering to capture everything
# file_transport = FileTransport(
#     filepath=log_file,
#     event_filter=None,  # Capture all events
# )

app = MCPApp(name="mcp_basic_agent")

async def example_usage():
    # Configure logging with our file transport
  #  await LoggingConfig.configure(transport=file_transport)
    
    async with app.run() as agent_app:
        logger = agent_app.logger
        context = agent_app.context

        logger.info("Current config:", data=context.config.model_dump())

        # Add the current directory to the filesystem server's args
        context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])

        finder_agent = Agent(
            name="finder",
            instruction="""You are an agent with access to the filesystem, 
            as well as the ability to fetch URLs. Your job is to identify 
            the closest match to a user's request, make the appropriate tool calls, 
            and return the URI and CONTENTS of the closest match.""",
            server_names=["fetch", "filesystem"],
        )

        async with finder_agent:
            logger.info("finder: Connected to server, calling list_tools...")
            result = await finder_agent.list_tools()
            logger.info("Tools available:", data=result.model_dump())

            llm = await finder_agent.attach_llm(OpenAIAugmentedLLM)
            result = await llm.generate_str(
                message="Print the contents of mcp_agent.config.yaml verbatim",
            )
            logger.info(f"Result: {result}")

            # Let's switch the same agent to a different LLM
            llm = await finder_agent.attach_llm(AnthropicAugmentedLLM)

            result = await llm.generate_str(
                message="Print the first 2 paragraphs of https://www.anthropic.com/research/building-effective-agents",
            )
            logger.info(f"Result: {result}")

            # Multi-turn conversations
            result = await llm.generate_str(
                message="Summarize those paragraphs in a 128 character tweet",
            )
            logger.info(f"Result: {result}")

    # Make sure to shut down logging cleanly
    await LoggingConfig.shutdown()


if __name__ == "__main__":
    start = time.time()
    asyncio.run(example_usage())
    end = time.time()
    t = end - start

    print(f"Total run time: {t:.2f}s")

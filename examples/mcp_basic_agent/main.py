import asyncio
import os

from mcp_agent.context import get_current_context
from mcp_agent.logging.logger import get_logger
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM


async def example_usage():
    logger = get_logger("mcp_basic_agent.example_usage")

    context = get_current_context()
    logger.info("Hello, world! Let's create an MCP Agent...")
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


if __name__ == "__main__":
    import time

    start = time.time()
    asyncio.run(example_usage())
    end = time.time()
    t = end - start

    print(f"Total run time: {t:.2f}s")

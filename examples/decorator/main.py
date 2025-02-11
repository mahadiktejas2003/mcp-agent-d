"""
Example MCP Agent application showing simplified agent access.
"""

import asyncio
from mcp_agent.core.decorator_app import MCPAgentDecorator

# Create the application
agent_app = MCPAgentDecorator("mcp-agent decorator example")


# Define the agent
# @agent_app.agent(
#     name="basic_agent",
#     instruction="A simple agent that helps with basic tasks.",
#     servers=["mcp_root"],
# )
@agent_app.agent(
    name="file_reader",
    instruction="An agent that can help with basic tasks, and access the filesystem.",
    servers=["filesystem"],
)
async def main():
    # Use the app's context manager - note we capture the yielded agent wrapper
    async with agent_app.run() as agent:
        result = await agent.send(
            "file_reader",
            "read the persona descriptions from 'persona.md' and rate their suitability for a junior PM job",
        )
        print(result)

        result = await agent.send(
            "file_reader",
            "can you write a letter to each candidate, informing them of the outcome of their application",
        )
        print(result)


if __name__ == "__main__":
    asyncio.run(main())

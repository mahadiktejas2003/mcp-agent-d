"""
Example MCP Agent application showing simplified agent access.
"""

import asyncio
from mcp_agent.core.decorator_app import MCPAgentDecorator
from rich import print

# Create the application
app = MCPAgentDecorator("root-test", config_path="mcp_agent.config.yaml")


# Define the agent
@app.agent(
    name="basic_agent",
    instruction="A simple agent that helps with basic tasks",
    servers=["mcp_root"],
)
async def main():
    # Use the app's context manager
    async with app.run() as agent_app:
        # Simple message exchange
        #    result = await agent_app.send("basic_agent", "count to 10")
        #   print(f"**************{result}")

        result = await agent_app.send("basic_agent", "what's the next  number?")
        print(f"------------{result}")


if __name__ == "__main__":
    asyncio.run(main())

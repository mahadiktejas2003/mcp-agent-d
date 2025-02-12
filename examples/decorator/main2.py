"""
Example MCP Agent application showing simplified agent access.
"""

import asyncio
from mcp_agent.core.decorator_app import MCPAgentDecorator

# Create the application
agent_app = MCPAgentDecorator("Decorator Analysis Example")


# Define the agent
# @agent_app.agent(
#     name="basic_agent",
#     instruction="A simple agent that helps with basic tasks.",
#     servers=["mcp_root"],
# )
@agent_app.agent(
    name="data_analysis",
    instruction="""
An advanced agent that can conduct sophisticated data analysis.
You have access to a Python interpreter, with pandas, matplotlib and seaborn installed. You can
install additional packages with the uv package manager.
You can search the internet, fetch URLs and access the filesystem.

You will be asked to interpet the data in three steps, so plan accordingly.
 - First, to get a basic understanding of the overall data.
 - Second, to conduct internet searches and downloads to gain additional insights that are useful for analysis.
 - Third, to conduct a final analysis of the data, presenting charts and a summary report.

""",
    servers=["filesystem", "interpreter", "brave", "fetch"],
)
async def main():
    # Use the app's context manager - note we capture the yielded agent wrapper
    async with agent_app.run() as agent:
        print(
            await agent(
                "use the python interpreter to conduct a basic analysis of the drivers, races and lap_times files in the current directory",
            )
        )
        print(
            await agent(
                "based on the data, use the internet search and fetch tools to gain additional insight on how to analyse the data",
            )
        )
        print(
            await agent(
                """
conduct a final detailed analysis of the data, using facts learned earlier. group data appropriately. 
produce relevant charts and reports (save charts to the current directory as .png files)
write a summary report, saved as 'summary.md'. if human input is required, ask for it. 
complete all tasks autonomously until the report and charts are completely saved.
"""
            )
        )


if __name__ == "__main__":
    asyncio.run(main())

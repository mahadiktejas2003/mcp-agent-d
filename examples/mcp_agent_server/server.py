from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio
import asyncio
from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

server = Server("mcp_agent")

app = MCPApp(name="mcp_server")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """
    List available tools.
    Each tool specifies its arguments using JSON Schema validation.
    """
    return [
        types.Tool(
            name="call_agent",
            description="Call an agent to generate a response",
            inputSchema={
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "description": "Name of an agent to call.",
                    },
                    "instruction": {
                        "type": "string",
                        "description": "Instructions for the agent",
                    },
                    "message": {
                        "type": "string",
                        "description": "Message to pass to an agent",
                    },
                    "server_names": {
                        "type": "array",
                        "description": "The MCP server to equip the agent with.",
                        "items": {
                            "type": "string",
                            "description": "The name of a MCP server.",
                        },
                        "uniqueItems": True,
                    },
                },
                "required": ["agent_name", "instruction", "message"],
            },
        ),
        types.Tool(
            name="list_servers",
            description="List all possible MCP servers that agent can use.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Handle tool execution requests.
    """
    if arguments is None:
        raise ValueError("Missing arguments")

    try:
        if name == "call_agent":
            await app.initialize()
            agent_name = arguments.get("agent_name")
            instruction = arguments.get("instruction")
            message = arguments.get("message")
            server_names = arguments.get("server_names")

            if not agent_name:
                raise ValueError("Missing agent_name parameter")
            elif not instruction:
                raise ValueError("Missing instruction parameter")
            elif not message:
                raise ValueError("Missing message parameter")

            agent = Agent(
                name=agent_name, instruction=instruction, server_names=server_names
            )

            await agent.initialize()

            llm = await agent.attach_llm(OpenAIAugmentedLLM)

            result = await llm.generate_str(
                message=message,
            )

            return [types.TextContent(type="text", text=result)]
        elif name == "list_servers":
            async with app.run() as agent_app:
                context = agent_app.context
                servers = list(context.config.mcp.servers.keys())
                return [types.TextContent(type="text", text="\n".join(servers))]
        else:
            ValueError(f"Unknown tool: {name}")

    except Exception as e:
        raise ValueError(f"Error: {e}")


async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="mcp_agent",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())

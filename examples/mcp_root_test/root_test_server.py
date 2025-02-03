from mcp.server.fastmcp import FastMCP, Context

mcp = FastMCP("MCP Root Tester")


@mcp.tool()
async def show_roots(ctx: Context) -> str:
    result = await ctx.session.list_roots()
    return result.roots


if __name__ == "__main__":
    mcp.run()

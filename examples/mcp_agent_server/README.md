# MCP Server example

To use in claude desktop, insert the following in `claude_desktop_config.json` and restart the Claude Desktop client.
```json
{
    "mcpServers": {
        "mcp-agent": {
            "command": "uv",
            "args": [
                "--directory",
                "/ABSOLUTE/PATH/TO/PARENT/FOLDER",
                "run",
                "server.py"
            ]
        }
    }
}
```

> [!WARNING]
> You may need to put the full path to the uv executable in the command field. You can get this by running `which uv` on MacOS/Linux or `where uv` on Windows.


To start the MCP client and server:
```bash
uv run client.py server.py
```
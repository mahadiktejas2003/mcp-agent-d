# marimo MCP Agent example

This example [marimo](https://github.com/marimo-team/marimo) notebook shows a
"finder" Agent which has access to the 'fetch' and 'filesystem' MCP servers.

You can ask it information about local files or URLs, and it will make the
determination on what to use at what time to satisfy the request.

First modify `mcp_agent.config.yaml` to include directories to which
you'd like to give the agent access.

Then run this example with:

```bash
OPENAI_API_KEY=<your-api-key> uvx marimo edit --sandbox notebook.py
```

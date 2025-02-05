# Contributing

We welcome **all** kinds of contributions - bug fixes, big features, docs, examples and more. _You don't need to be an AI expert
or even a Python developer to help out._

## Checklist

Contributions are made through
[pull requests](https://help.github.com/articles/using-pull-requests/).

Before sending a pull request, make sure to do the following:

- Fork the repo, and create a feature branch prefixed with `feature/`
- [Lint, typecheck, and format](#lint-typecheck-format) your code
- [Add examples](#examples)

_Please reach out to the mcp-agent maintainers before starting work on a large
contribution._ Get in touch at
[GitHub issues](https://github.com/lastmile-ai/mcp-agent/issues)
or [on Discord](https://lmai.link/discord/mcp-agent).

## Prerequisites

To build mcp-agent, you'll need the following installed:

- Install [uv](https://docs.astral.sh/uv/), which we use for Python package management
- Install [Python](https://www.python.org/) >= 3.10. (You may already it installed. To see your version, use `python -V` at the command line.)

  If you don't, install it using `uv python install 3.10`

- Install dev dependencies using `uv sync --dev`

## Scripts

There are several useful scripts in the `scripts/` directory that can be invoked via `uv run scripts/<script>.py [ARGS]`

### promptify.py

Bundles the mcp-agent repo into a single `project_contents.md` so you can use it as a prompt for LLMs to help you develop.
Use `-i REGEX` to include only specific files, and `-x REGEX` to exclude certain files.

Example:

```bash
uv run scripts/promptify.py -i "**/agents/**" -i "**/context.py" -x "**/app.py"
```

### example.py

This script lets you run any example in the `examples/` directory in debug mode. It configures the venv for the example,
installs its dependencies from `requirements.txt`, and runs the example.

To run:

```bash
uv run scripts/example.py run <example_name> --debug
```

To clean:

```bash
uv run scripts/example.py clean <example_name>
```

Example usage to run `examples/workflow_orchestrator_worker`:

```bash
uv run scripts/example.py run workflow_orchestrator_worker --debug
```

## Lint, Typecheck, Format

Lint and format is run as part of the precommit hook defined in [.pre-commit-config.yaml](./.pre-commit-config.yaml).

**Lint**

```bash
uv run scripts/lint.py --fix
```

**Format**

```bash
uv run scripts/format.py
```

## Examples

We use the examples for end-to-end testing. We'd love for you to add Python unit tests for new functionality going forward.

At minimum, for any new feature or provider integration (e.g. additional LLM support), you should add example usage in the [`examples`](./examples/) directory.

## Editor settings

If you use vscode, you might find the following `settings.json` useful. We've added them to the [.vscode](./.vscode) directory along with recommended extensions

```json
{
  "editor.formatOnSave": true,
  "editor.defaultFormatter": "esbenp.prettier-vscode",
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "editor.rulers": []
  },
  "yaml.schemas": {
    "https://raw.githubusercontent.com/lastmile-ai/mcp-agent/main/schema/mcp-agent.config.schema.json": [
      "mcp-agent.config.yaml",
      "mcp_agent.config.yaml",
      "mcp-agent.secrets.yaml",
      "mcp_agent.secrets.yaml"
    ]
  }
}
```

## Thank you

If you are considering contributing, or have already done so, **thank you**. This project is meant to streamline AI application development, and we need all the help we can get! Happy building.

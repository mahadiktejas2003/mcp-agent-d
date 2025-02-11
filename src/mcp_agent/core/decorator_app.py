"""
Decorator-based interface for MCP Agent applications.
Provides a simplified way to create and manage agents using decorators.
"""

from typing import List, Optional, Any, Dict, Callable, TypeVar, AsyncIterator
import logging
import yaml
from contextlib import asynccontextmanager

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.config import Settings

# Set up logging
logger = logging.getLogger(__name__)

# Type variables for better type hints
T = TypeVar("T")
AgentConfig = Dict[str, Dict[str, Any]]


class MCPAgentDecorator:
    """
    A decorator-based interface for MCP Agent applications.
    Provides a simplified way to create and manage agents using decorators.

    Example:
        agent_app = MCPAgentDecorator("my-app")

        @agent_app.agent(
            name="basic_agent",
            instruction="A simple agent that helps with basic tasks.",
            servers=["mcp_root"],
        )
        async def main():
            async with agent_app.run() as agent:
                result = await agent.send("basic_agent", "Hello!")
                print(result)
    """

    def __init__(self, name: str, config_path: Optional[str] = None) -> None:
        """
        Initialize the decorator interface.

        Args:
            name: Name of the application
            config_path: Optional path to config file

        Raises:
            FileNotFoundError: If config_path is provided but file doesn't exist
            yaml.YAMLError: If config file contains invalid YAML
        """
        self.name = name
        self.config_path = config_path
        self._load_config()
        self.app = MCPApp(
            name=name,
            settings=Settings(**self.config) if hasattr(self, "config") else None,
        )
        self.agents: AgentConfig = {}

    def _load_config(self) -> None:
        """
        Load configuration from YAML file.

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file contains invalid YAML
        """
        if self.config_path:
            try:
                with open(self.config_path) as f:
                    self.config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                logger.error(f"Failed to parse config file: {e}")
                raise
            except FileNotFoundError:
                logger.error(f"Config file not found: {self.config_path}")
                raise

    def agent(
        self, name: str, instruction: str, servers: List[str]
    ) -> Callable[[T], T]:
        """
        Decorator to create and register an agent.

        Args:
            name: Name of the agent
            instruction: Base instruction for the agent
            servers: List of server names the agent should connect to

        Returns:
            Decorator function that wraps the original async function

        Raises:
            ValueError: If name is already registered
        """
        if name in self.agents:
            raise ValueError(f"Agent '{name}' is already registered")

        def decorator(func: T) -> T:
            self.agents[name] = {"instruction": instruction, "servers": servers}
            return func

        return decorator

    @asynccontextmanager
    async def run(self) -> AsyncIterator["AgentAppWrapper"]:
        """
        Context manager for running the application.
        Handles setup and teardown of the app and agents.

        Yields:
            AgentAppWrapper instance for interacting with agents

        Raises:
            RuntimeError: If agent initialization fails
        """
        async with self.app.run() as agent_app:
            active_agents: Dict[str, Agent] = {}
            agent_contexts: List[Agent] = []

            try:
                # Create and initialize all registered agents
                for name, config in self.agents.items():
                    try:
                        agent = Agent(
                            name=name,
                            instruction=config["instruction"],
                            server_names=config["servers"],
                            context=agent_app.context,
                        )
                        active_agents[name] = agent

                        # Start agent within context manager
                        await agent.__aenter__()
                        agent_contexts.append(agent)

                        # Attach LLM to agent
                        llm = await agent.attach_llm(AnthropicAugmentedLLM)
                        agent._llm = llm

                    except Exception as e:
                        logger.error(f"Failed to initialize agent {name}: {e}")
                        # Clean up any agents that were initialized
                        await self._cleanup_agents(agent_contexts)
                        raise RuntimeError(f"Agent initialization failed: {e}") from e

                wrapper = AgentAppWrapper(agent_app, active_agents)
                yield wrapper

            finally:
                await self._cleanup_agents(agent_contexts)

    @staticmethod
    async def _cleanup_agents(agents: List[Agent]) -> None:
        """
        Clean up agents in reverse order of creation.

        Args:
            agents: List of agents to clean up
        """
        for agent in reversed(agents):
            try:
                await agent.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"Error cleaning up agent {agent.name}: {e}")


class AgentAppWrapper:
    """
    Wrapper class providing a simplified interface to the agent application.
    """

    def __init__(self, app: MCPApp, agents: Dict[str, Agent]) -> None:
        """
        Initialize the wrapper.

        Args:
            app: The MCPApp instance
            agents: Dictionary of agent name to Agent instance
        """
        self.app = app
        self.agents = agents

    async def send(self, agent_name: str, message: str) -> str:
        """
        Send a message to a specific agent and get the response.

        Args:
            agent_name: Name of the agent to send message to
            message: Message to send

        Returns:
            Agent's response as a string

        Raises:
            ValueError: If agent_name is not found
            RuntimeError: If agent has no LLM attached
        """
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' not found")

        agent = self.agents[agent_name]
        if not hasattr(agent, "_llm") or agent._llm is None:
            raise RuntimeError(f"Agent '{agent_name}' has no LLM attached")

        return await agent._llm.generate_str(message)

    async def __call__(self, message: str) -> str:
        return self.send(self, self.agents[0].name, message)

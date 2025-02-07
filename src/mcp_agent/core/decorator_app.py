"""
Decorator-based application class for MCP Agent.
Provides a clean, pythonic interface for creating MCP agents and workflows.
"""

import asyncio
import functools
from typing import Any, Callable, Dict, List, Optional, Type, Union

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm import AugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.config import Settings


class MCPAgentDecorator:
    """
    A decorator-based interface for creating MCP agents and workflows.
    Provides a clean, pythonic way to define agents and their behaviors.
    """
    
    def __init__(self, name: str, config_path: Optional[str] = None):
        """
        Initialize the MCP Agent application.
        
        Args:
            name: Name of the application
            config_path: Optional path to configuration file
        """
        self.name = name
        self.config_path = config_path
        settings = Settings(_env_file=config_path) if config_path else None
        self.app = MCPApp(name=name, settings=settings)
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.workflows: List[Dict[str, Any]] = []
        self._default_llm = OpenAIAugmentedLLM

    def agent(
        self, 
        name: str, 
        servers: Optional[List[str]] = None,
        instruction: Optional[str] = None,
        llm_class: Optional[Type[AugmentedLLM]] = None
    ):
        """
        Decorator to create an agent.
        
        Args:
            name: Name of the agent
            servers: List of server names the agent can access
            instruction: Agent's instruction/system prompt
            llm_class: Optional LLM class to use (defaults to OpenAI)
        """
        def decorator(func: Callable):
            self.agents[name] = {
                "function": func,
                "servers": servers or [],
                "instruction": instruction or func.__doc__ or "",
                "llm": llm_class or self._default_llm
            }
            
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                return await func(*args, **kwargs)
            return wrapper
        return decorator

    def llm(self, llm_class: Type[AugmentedLLM]):
        """
        Decorator to specify LLM for an agent.
        
        Args:
            llm_class: The LLM class to use for this agent
        """
        def decorator(func: Callable):
            agent_name = func.__name__
            if agent_name in self.agents:
                self.agents[agent_name]["llm"] = llm_class
            
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                return await func(*args, **kwargs)
            return wrapper
        return decorator

    async def run_agent(self, agent_name: str, *args, **kwargs) -> Any:
        """
        Run a specific agent.
        
        Args:
            agent_name: Name of the agent to run
            *args: Arguments to pass to the agent
            **kwargs: Keyword arguments to pass to the agent
        
        Returns:
            The result of the agent's execution
        """
        if agent_name not in self.agents:
            raise ValueError(f"Agent {agent_name} not found")

        agent_config = self.agents[agent_name]
        
        agent = Agent(
            name=agent_name,
            instruction=agent_config["instruction"],
            server_names=agent_config["servers"]
        )

        async with agent:
            llm_class = agent_config.get("llm", self._default_llm)
            llm = await agent.attach_llm(llm_class)
            result = await agent_config["function"](llm, *args, **kwargs)
            return result

    def run(self):
        """Get the application context manager for running the app"""
        return self.app.run()
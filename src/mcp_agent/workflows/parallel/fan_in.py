import contextlib
from typing import Callable, Dict, List, Optional, Type, TYPE_CHECKING

from mcp_agent.agents.agent import Agent
from mcp_agent.context_dependent import ContextDependent
from mcp_agent.workflows.llm.augmented_llm import (
    AugmentedLLM,
    MessageParamT,
    MessageT,
    ModelT,
)

if TYPE_CHECKING:
    from mcp_agent.context import Context

FanInInput = (
    # Dict of agent/source name to list of messages generated by that agent
    Dict[str, List[MessageT] | List[MessageParamT]]
    # Dict of agent/source name to string generated by that agent
    | Dict[str, str]
    # List of lists of messages generated by each agent
    | List[List[MessageT] | List[MessageParamT]]
    # List of strings generated by each agent
    | List[str]
)


class FanIn(ContextDependent):
    """
    Aggregate results from multiple parallel tasks into a single result.

    This is a building block of the Parallel workflow, which can be used to fan out
    work to multiple agents or other parallel tasks, and then aggregate the results.

    For example, you can use FanIn to combine the results of multiple agents into a single response,
    such as a Summarization Fan-In agent that combines the outputs of multiple language models.
    """

    def __init__(
        self,
        aggregator_agent: Agent | AugmentedLLM[MessageParamT, MessageT],
        llm_factory: Callable[[Agent], AugmentedLLM[MessageParamT, MessageT]] = None,
        context: Optional["Context"] = None,
        **kwargs,
    ):
        """
        Initialize the FanIn with an Agent responsible for processing multiple responses into a single aggregated one.
        """

        super().__init__(context=context, **kwargs)

        self.executor = self.context.executor
        self.llm_factory = llm_factory
        self.aggregator_agent = aggregator_agent

        if not isinstance(self.aggregator_agent, AugmentedLLM):
            if not self.llm_factory:
                raise ValueError("llm_factory is required when using an Agent")

    async def generate(
        self,
        messages: FanInInput,
        use_history: bool = True,
        max_iterations: int = 10,
        model: str = None,
        stop_sequences: List[str] = None,
        max_tokens: int = 2048,
        parallel_tool_calls: bool = True,
    ) -> List[MessageT]:
        """
        Request fan-in agent generation from a list of messages from multiple sources/agents.
        Internally aggregates the messages and then calls the aggregator agent to generate a response.
        """
        message: (
            str | MessageParamT | List[MessageParamT]
        ) = await self.aggregate_messages(messages)

        async with contextlib.AsyncExitStack() as stack:
            if isinstance(self.aggregator_agent, AugmentedLLM):
                llm = self.aggregator_agent
            else:
                # Enter agent context
                ctx_agent = await stack.enter_async_context(self.aggregator_agent)
                llm = await ctx_agent.attach_llm(self.llm_factory)

            return await llm.generate(
                message=message,
                use_history=use_history,
                max_iterations=max_iterations,
                model=model,
                stop_sequences=stop_sequences,
                max_tokens=max_tokens,
                parallel_tool_calls=parallel_tool_calls,
            )

    async def generate_str(
        self,
        messages: FanInInput,
        use_history: bool = True,
        max_iterations: int = 10,
        model: str = None,
        stop_sequences: List[str] = None,
        max_tokens: int = 2048,
        parallel_tool_calls: bool = True,
    ) -> str:
        """
        Request fan-in agent generation from a list of messages from multiple sources/agents.
        Internally aggregates the messages and then calls the aggregator agent to generate a
        response, which is returned as a string.
        """

        message: (
            str | MessageParamT | List[MessageParamT]
        ) = await self.aggregate_messages(messages)

        async with contextlib.AsyncExitStack() as stack:
            if isinstance(self.aggregator_agent, AugmentedLLM):
                llm = self.aggregator_agent
            else:
                # Enter agent context
                ctx_agent = await stack.enter_async_context(self.aggregator_agent)
                llm = await ctx_agent.attach_llm(self.llm_factory)

            return await llm.generate_str(
                message=message,
                use_history=use_history,
                max_iterations=max_iterations,
                model=model,
                stop_sequences=stop_sequences,
                max_tokens=max_tokens,
                parallel_tool_calls=parallel_tool_calls,
            )

    async def generate_structured(
        self,
        messages: FanInInput,
        response_model: Type[ModelT],
        use_history: bool = True,
        max_iterations: int = 10,
        model: str = None,
        stop_sequences: List[str] = None,
        max_tokens: int = 2048,
        parallel_tool_calls: bool = True,
    ) -> ModelT:
        """
        Request a structured fan-in agent generation from a list of messages
        from multiple sources/agents. Internally aggregates the messages and then calls
        the aggregator agent to generate a response, which is returned as a Pydantic model.
        """

        message: (
            str | MessageParamT | List[MessageParamT]
        ) = await self.aggregate_messages(messages)

        async with contextlib.AsyncExitStack() as stack:
            if isinstance(self.aggregator_agent, AugmentedLLM):
                llm = self.aggregator_agent
            else:
                # Enter agent context
                ctx_agent = await stack.enter_async_context(self.aggregator_agent)
                llm = await ctx_agent.attach_llm(self.llm_factory)

            return await llm.generate_structured(
                message=message,
                response_model=response_model,
                use_history=use_history,
                max_iterations=max_iterations,
                model=model,
                stop_sequences=stop_sequences,
                max_tokens=max_tokens,
                parallel_tool_calls=parallel_tool_calls,
            )

    async def aggregate_messages(
        self, messages: FanInInput
    ) -> str | MessageParamT | List[MessageParamT]:
        """
        Aggregate messages from multiple sources/agents into a single message to
        use with the aggregator agent generation.

        The input can be a dictionary of agent/source name to list of messages
        generated by that agent, or just the unattributed lists of messages to aggregate.

        Args:
            messages: Can be one of:
                - Dict[str, List[MessageT] | List[MessageParamT]]: Dict of agent names to messages
                - Dict[str, str]: Dict of agent names to message strings
                - List[List[MessageT] | List[MessageParamT]]: List of message lists from agents
                - List[str]: List of message strings from agents

        Returns:
            Aggregated message as string, MessageParamT or List[MessageParamT]

        Raises:
            ValueError: If input is empty or contains empty/invalid elements
        """
        # Handle dictionary inputs
        if isinstance(messages, dict):
            # Check for empty dict
            if not messages:
                raise ValueError("Input dictionary cannot be empty")

            first_value = next(iter(messages.values()))

            # Dict[str, List[MessageT] | List[MessageParamT]]
            if isinstance(first_value, list):
                if any(not isinstance(v, list) for v in messages.values()):
                    raise ValueError("All dictionary values must be lists of messages")
                # Process list of messages for each agent
                return await self.aggregate_agent_messages(messages)

            # Dict[str, str]
            elif isinstance(first_value, str):
                if any(not isinstance(v, str) for v in messages.values()):
                    raise ValueError("All dictionary values must be strings")
                # Process string outputs from each agent
                return await self.aggregate_agent_message_strings(messages)

            else:
                raise ValueError(
                    "Dictionary values must be either lists of messages or strings"
                )

        # Handle list inputs
        elif isinstance(messages, list):
            # Check for empty list
            if not messages:
                raise ValueError("Input list cannot be empty")

            first_item = messages[0]

            # List[List[MessageT] | List[MessageParamT]]
            if isinstance(first_item, list):
                if any(not isinstance(item, list) for item in messages):
                    raise ValueError("All list items must be lists of messages")
                # Process list of message lists
                return await self.aggregate_message_lists(messages)

            # List[str]
            elif isinstance(first_item, str):
                if any(not isinstance(item, str) for item in messages):
                    raise ValueError("All list items must be strings")
                # Process list of strings
                return await self.aggregate_message_strings(messages)

            else:
                raise ValueError(
                    "List items must be either lists of messages or strings"
                )

        else:
            raise ValueError(
                "Input must be either a dictionary of agent messages or a list of messages"
            )

    # Helper methods for processing different types of inputs
    async def aggregate_agent_messages(
        self, messages: Dict[str, List[MessageT] | List[MessageParamT]]
    ) -> str | MessageParamT | List[MessageParamT]:
        """
        Aggregate message lists with agent names.

        Args:
            messages: Dictionary mapping agent names to their message lists

        Returns:
            str | List[MessageParamT]: Messages formatted with agent attribution

        """

        # In the default implementation, we'll just convert the messages to a
        # single string with agent attribution
        aggregated_messages = []

        if not messages:
            return ""

        # Format each agent's messages with attribution
        for agent_name, agent_messages in messages.items():
            agent_message_strings = []
            for msg in agent_messages or []:
                if isinstance(msg, str):
                    agent_message_strings.append(f"Agent {agent_name}: {msg}")
                else:
                    # Assume it's a Message/MessageParamT and add attribution
                    agent_message_strings.append(f"Agent {agent_name}: {str(msg)}")

            aggregated_messages.append("\n".join(agent_message_strings))

        # Combine all messages with clear separation
        final_message = "\n\n".join(aggregated_messages)
        final_message = f"Aggregated responses from multiple Agents:\n\n{final_message}"
        return final_message

    async def aggregate_agent_message_strings(self, messages: Dict[str, str]) -> str:
        """
        Aggregate string outputs with agent names.

        Args:
            messages: Dictionary mapping agent names to their string outputs

        Returns:
            str: Combined string with agent attributions
        """
        if not messages:
            return ""

        # Format each agent's message with agent attribution
        aggregated_messages = [
            f"Agent {agent_name}: {message}" for agent_name, message in messages.items()
        ]

        # Combine all messages with clear separation
        final_message = "\n\n".join(aggregated_messages)
        final_message = f"Aggregated responses from multiple Agents:\n\n{final_message}"
        return final_message

    async def aggregate_message_lists(
        self, messages: List[List[MessageT] | List[MessageParamT]]
    ) -> str | MessageParamT | List[MessageParamT]:
        """
        Aggregate message lists without agent names.

        Args:
            messages: List of message lists from different agents

        Returns:
            List[MessageParamT]: List of formatted messages
        """
        aggregated_messages = []

        if not messages:
            return ""

        # Format each source's messages
        for i, source_messages in enumerate(messages, 1):
            source_message_strings = []
            for msg in source_messages or []:
                if isinstance(msg, str):
                    source_message_strings.append(f"Source {i}: {msg}")
                else:
                    # Assume it's a MessageParamT or MessageT and add source attribution
                    source_message_strings.append(f"Source {i}: {str(msg)}")

            aggregated_messages.append("\n".join(source_messages))

        # Combine all messages with clear separation
        final_message = "\n\n".join(aggregated_messages)
        final_message = (
            f"Aggregated responses from multiple sources:\n\n{final_message}"
        )
        return final_message

    async def aggregate_message_strings(self, messages: List[str]) -> str:
        """
        Aggregate string outputs without agent names.

        Args:
            messages: List of string outputs from different agents

        Returns:
            str: Combined string with source attributions
        """
        if not messages:
            return ""

        # Format each source's message with attribution
        aggregated_messages = [
            f"Source {i}: {message}" for i, message in enumerate(messages, 1)
        ]

        # Combine all messages with clear separation
        final_message = "\n\n".join(aggregated_messages)
        final_message = (
            f"Aggregated responses from multiple sources:\n\n{final_message}"
        )
        return final_message

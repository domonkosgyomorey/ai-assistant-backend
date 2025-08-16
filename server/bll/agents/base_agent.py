from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Generator

from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import Runnable


class BaseAgent(ABC):
    """
    Abstract base class for all agents that handles common invoke patterns.
    Abstracts away invoke, ainvoke, stream, astream to avoid code duplication.
    """

    def __init__(self, llm: BaseLanguageModel, verbose: bool = False):
        self.llm = llm
        self.verbose = verbose
        self.chain: Runnable = self._build_chain()

    @abstractmethod
    def _build_chain(self) -> Runnable:
        """Build the LangChain Runnable chain for this agent."""
        ...

    def _log(self, message: str, data: Any = None):
        """Simple logging utility."""
        if self.verbose:
            if data is not None:
                print(f"[{self.__class__.__name__}] {message}: {data}")
            else:
                print(f"[{self.__class__.__name__}] {message}")

    def invoke(self, query: str, **kwargs) -> str:
        """
        Synchronous invocation of the agent.

        Args:
            query: The input query
            **kwargs: Additional parameters

        Returns:
            The agent's response as a string
        """
        try:
            input_dict = {"question": query, **kwargs}
            result = self.chain.invoke(input_dict)
            return str(result)
        except Exception as e:
            self._log(f"Error in invoke: {str(e)}")
            return f"I encountered an error while processing your question: {str(e)}"

    async def ainvoke(self, query: str, **kwargs) -> str:
        """
        Asynchronous invocation of the agent.

        Args:
            query: The input query
            **kwargs: Additional parameters

        Returns:
            The agent's response as a string
        """
        try:
            input_dict = {"question": query, **kwargs}
            result = await self.chain.ainvoke(input_dict)
            return str(result)
        except Exception as e:
            self._log(f"Error in ainvoke: {str(e)}")
            return f"I encountered an error while processing your question: {str(e)}"

    def stream(self, query: str, **kwargs) -> Generator[str, None, None]:
        """
        Stream the agent's response.

        Args:
            query: The input query
            **kwargs: Additional parameters

        Yields:
            Chunks of the agent's response
        """
        try:
            input_dict = {"question": query, **kwargs}
            for chunk in self.chain.stream(input_dict):
                yield str(chunk)
        except Exception as e:
            self._log(f"Error in stream: {str(e)}")
            yield f"Error: {str(e)}"

    async def astream(self, query: str, **kwargs) -> AsyncGenerator[str, None]:
        """
        Asynchronously stream the agent's response.

        Args:
            query: The input query
            **kwargs: Additional parameters

        Yields:
            Chunks of the agent's response
        """
        try:
            input_dict = {"question": query, **kwargs}
            async for chunk in self.chain.astream(input_dict):
                yield str(chunk)
        except Exception as e:
            self._log(f"Error in astream: {str(e)}")
            yield f"Error: {str(e)}"

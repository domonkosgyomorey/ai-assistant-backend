from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Generator

from core.config.config import config
from langchain_core.language_models import BaseLanguageModel


class BaseAgent(ABC):
    """
    Abstract base class for all agents that handles common invoke patterns.
    Abstracts away invoke, ainvoke, stream, astream to avoid code duplication.
    """

    def __init__(self, llm: BaseLanguageModel, verbose: bool = False):
        self.llm = llm
        self.verbose = verbose
        self.chain = self._build_chain()

    @abstractmethod
    def _build_chain(self):
        """Build the LangChain Runnable chain for this agent."""
        ...

    def _log(self, message: str, data: Any = None):
        """Simple logging utility."""
        if self.verbose:
            if data is not None:
                print(f"[{self.__class__.__name__}] {message}: {data}")
            else:
                print(f"[{self.__class__.__name__}] {message}")

    def invoke(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Synchronous invocation of the agent.

        Args:
            input_data: The input dictionary

        Returns:
            The agent's response as a dictionary
        """
        try:
            result = self.chain.invoke(input_data, config={"callbacks": [config.langfuse_handler]})
            if isinstance(result, dict):
                return result
            return {"content": str(result)}
        except Exception as e:
            self._log(f"Error in invoke: {str(e)}")
            return {"error": str(e), "content": f"I encountered an error while processing your request: {str(e)}"}

    async def ainvoke(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Asynchronous invocation of the agent.

        Args:
            input_data: The input dictionary

        Returns:
            The agent's response as a dictionary
        """
        try:
            result = await self.chain.ainvoke(input_data, config={"callbacks": [config.langfuse_handler]})
            if isinstance(result, dict):
                return result
            return {"content": str(result)}
        except Exception as e:
            self._log(f"Error in ainvoke: {str(e)}")
            return {"error": str(e), "content": f"I encountered an error while processing your request: {str(e)}"}

    def stream(self, input_data: dict[str, Any]) -> Generator[dict[str, Any], None, None]:
        """
        Stream the agent's response.

        Args:
            input_data: The input dictionary

        Yields:
            Chunks of the agent's response as dictionaries
        """
        try:
            for chunk in self.chain.stream(input_data, config={"callbacks": [config.langfuse_handler]}):
                if isinstance(chunk, dict):
                    yield chunk
                else:
                    yield {"content": str(chunk)}
        except Exception as e:
            self._log(f"Error in stream: {str(e)}")
            yield {"error": str(e), "content": f"Error: {str(e)}"}

    async def astream(self, input_data: dict[str, Any]) -> AsyncGenerator[dict[str, Any], None]:
        """
        Asynchronously stream the agent's response.

        Args:
            input_data: The input dictionary

        Yields:
            Chunks of the agent's response as dictionaries
        """
        try:
            async for chunk in self.chain.astream(input_data, config={"callbacks": [config.langfuse_handler]}):
                if isinstance(chunk, dict):
                    yield chunk
                else:
                    yield {"content": str(chunk)}
        except Exception as e:
            self._log(f"Error in astream: {str(e)}")
            yield {"error": str(e), "content": f"Error: {str(e)}"}

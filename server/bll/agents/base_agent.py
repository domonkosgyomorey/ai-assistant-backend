import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, Optional

from langchain_core.language_models import BaseChatModel


@dataclass
class AgentResponse:
    text: str
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)


class BaseAgent:
    def __init__(self, chatllm: BaseChatModel):
        self.chatllm = chatllm

    async def ainvoke(self, prompt: str) -> AgentResponse:
        raise NotImplementedError()

    async def astream(self, prompt: str) -> AsyncGenerator[AgentResponse, None]:
        response = await self.ainvoke(prompt)
        yield response

    def invoke(self, prompt: str) -> AgentResponse:
        return asyncio.run(self.ainvoke(prompt))

    def stream(self, prompt: str):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        agen = self.astream(prompt)
        try:
            while True:
                chunk = loop.run_until_complete(agen.__anext__())
                yield chunk
        except StopAsyncIteration:
            pass
        finally:
            loop.close()

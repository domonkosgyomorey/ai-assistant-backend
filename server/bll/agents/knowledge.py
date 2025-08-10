from typing import AsyncGenerator, List

from bll.agents.base_agent import AgentResponse, BaseAgent
from bll.agents.pompts import KNOWLEDGE_PROMPT
from langchain.schema import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import RetrieverLike


class KnowledgeAgent(BaseAgent):
    def __init__(
        self,
        chatllm: BaseChatModel,
        retriever: RetrieverLike,
        agent_description: str,
    ):
        super().__init__(chatllm)
        self.knowledge_retriever = retriever
        self.agent_description = agent_description
        self.prompt_template = KNOWLEDGE_PROMPT

    def format_prompt(self, prompt: str, docs: List[Document]) -> str:
        return self.prompt_template.format(
            about_you=self.agent_description, context="\n".join([doc.page_content for doc in docs]), user_prompt=prompt
        )

    async def ainvoke(self, prompt: str) -> AgentResponse:
        docs = await self.knowledge_retriever.ainvoke(prompt)

        full_prompt = self.format_prompt(prompt, docs)
        text = await self.chatllm.ainvoke(full_prompt)
        metadata = {"retrieved_docs": docs, "agent_description": self.agent_description}
        return AgentResponse(text=text, metadata=metadata)

    async def astream(self, prompt: str) -> AsyncGenerator[AgentResponse, None]:
        docs = await self.knowledge_retriever.ainvoke(prompt)
        full_prompt = self.format_prompt(prompt, docs)

        async for token in self.chatllm.astream(full_prompt):
            yield AgentResponse(text=token)

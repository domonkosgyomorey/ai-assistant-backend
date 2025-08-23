from core.utils.components import get_llm
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from bll.agents.base_agent import BaseAgent
from bll.agents.contextualizer_agent.prompt import CONTEXTUALIZER_PROMPT


class ContextualizerAgent(BaseAgent):
    def __init__(self):
        super().__init__(get_llm(), verbose=True)

    def _build_chain(self):
        return RunnablePassthrough.assign(contextual_prompt=CONTEXTUALIZER_PROMPT | self.llm | StrOutputParser())

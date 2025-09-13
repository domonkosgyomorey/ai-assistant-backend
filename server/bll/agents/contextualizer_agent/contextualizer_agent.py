from core.utils.components import get_llm
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from bll.agents.base_agent import BaseAgent
from bll.agents.contextualizer_agent.prompt import CONTEXTUALIZER_PROMPT


class ContextualizerAgent(BaseAgent):
    def __init__(self):
        super().__init__(get_llm(), verbose=True)

    def _contextualize(self, input_dict) -> str:
        formatted_prompt = CONTEXTUALIZER_PROMPT.format_prompt(**input_dict)
        return StrOutputParser().invoke(self.llm.invoke(formatted_prompt))

    def _process_input(self, input_dict) -> str:
        return input_dict["prompt"] if len(input_dict.get("history", "")) == 0 else self._contextualize(input_dict)

    def _build_chain(self):
        return RunnablePassthrough.assign(contextual_prompt=self._process_input)

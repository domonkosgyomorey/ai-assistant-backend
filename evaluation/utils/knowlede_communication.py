import os
import sys

sys.path.append(os.path.join(os.getcwd(), "server"))

from bll.agents.knowledge import Knowledge
from langchain_core.messages import AIMessage, HumanMessage

knowledge_agent = Knowledge()


def call(question: str, history: list[dict[str, str]] | None = None) -> str:
    messages = []

    if history:
        for msg in history:
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg.get("role") == "ai":
                messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=question))

    try:
        result = knowledge_agent.invoke({"messages": messages})

        if isinstance(result, dict):
            answer = result.get("answer", result.get("content", str(result)))
        else:
            answer = str(result)

        return answer

    except Exception as e:
        print(f"Error calling knowledge agent: {e}")
        return f"Error: Could not process the question due to: {str(e)}"

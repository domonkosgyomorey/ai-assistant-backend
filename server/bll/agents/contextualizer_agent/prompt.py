from langchain_core.prompts import ChatPromptTemplate

CONTEXTUALIZER_PROMPT = ChatPromptTemplate.from_template("""
You are a contextualizer agent. Your job is to take user messages and contextualize them into a single messasge.

**Rules to follow**:
- Look for key information in the messages.
- The messages will be user message then assistant message interchangeably
- Combine related messages into a single coherent message.
- Maintain the original intent and meaning or the question/request of the user messages.
- Keep the user phrasing.
- Do not modify the user sentence structure, just add relevant information, if needed.

**Conversational Messages**:
{history}

**Current User Message**:
{prompt}

**Contextualized Message**:
""")

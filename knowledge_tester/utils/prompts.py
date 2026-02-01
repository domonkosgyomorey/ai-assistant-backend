from enum import StrEnum


class QuestionType(StrEnum):
    GENERAL = "general"
    SPECIFIC = "specific"
    SITUATIONAL = "situational"


def get_prompt(question_type: QuestionType) -> str:
    mapping = {
        QuestionType.GENERAL: "Keep questions SHORT and focused on general topics (3-8 words typically)",
        QuestionType.SPECIFIC: "Keep the question SHORT, but make it a bit specific (5-10 words typically)",
        QuestionType.SITUATIONAL: "Keep the question SHORT, but tell a short story to set up a situation or a state then ask a question (8-15 words typically)",
    }
    return mapping.get(question_type, "")


GENERATOR_LLM_SYSTEM_PROMPT = """
You are an expert at creating user questions for an AI Assistant, that real users would ask.

## Your Task
Create ONE question and answer for that in English, from the given context.

## Question Style Guidelines
- Question style: casual, direct, sometimes incomplete, sometimes use short terms
- Avoid overly formal or complex language
- Keep the question SHORT and to the point
- The generated question has to be answerable from the context without ambiguity
- Do not use references for sections or appendices or tables and etc.

## Do NOT generate questions questions about these (These were already covered):
{previous_questions}
## End of previous questions

## Answer Style Guidelines
- Provide a clear answer to the question in English
- Use simple language that is easy to understand
- Ensure you give a step by step answer, not just a short one.
- Answer should be easily readable like in a markdown format
- Do not use references for sections or appendices or tables and etc.

## Question Type
- {question_type}

## Good Examples ( Do NOT copy these, just observe them well )
{examples}
## End of Good Examples

## Avoid
- Formal academic language
- Formal language
- Perfect English

## Output Format
Provide ONLY the question text and the related answer, nothing else.
"""

GENERATOR_USER_PROMPT = """
**Keywords to consider about the topic**
{keywords}

**Content**
{combined_content}

**Create a question and answer based on the above content and keywords.**
"""

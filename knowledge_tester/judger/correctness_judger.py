from typing import Optional

from judger.base_judger import BaseJudger
from pydantic import BaseModel, Field
from utils.common_types import AnsweredTestCase
from utils.components import llm


class JudgementOutput(BaseModel):
    factual_correctness: bool = Field(
        ...,
        description="True if the system answer is factually correct when compared to the reference answer, False otherwise.",
    )
    completeness: int = Field(
        ...,
        description="An integer score (0-5) indicating how complete the system answer is. 0 means missing all key information, 5 means fully complete with all important points covered.",
    )
    proof: str = Field(
        ...,
        description="Detailed explanation justifying the scores. Explain what is correct, what is missing, or what is incorrect. If fully correct and complete, briefly confirm this.",
    )
    extra_information: str = Field(
        ...,
        description="Description of any additional information present in the system answer that is not in the reference answer. If none, this should be empty or 'N/A'.",
    )


class AggregatedJudgement(BaseModel):
    average_factual_correctness: float = Field(
        ...,
        description="Average factual correctness score across multiple judgements, represented as a float between 0 and 1.",
    )
    average_completeness: float = Field(
        ...,
        description="Average completeness score across multiple judgements, represented as a float between 0 and 5.",
    )
    average_extra_information: float = Field(
        ...,
        description="Average extra information score across multiple judgements, represented as a float between 0 and 1.",
    )


JUDGER_SYSTEM_PROMPT = """You are an expert evaluator assessing the quality of AI-generated answers.

Your task is to evaluate the system's answer using three independent criteria:

## 1. Factual Correctness (True/False)
**IMPORTANT**: This is NOT about completeness. Mark as True if:
- The system answer contains NO factual contradictions or errors
- Even if the answer is brief or incomplete, as long as what IS stated is correct, mark True
- You are checking for wrong information, and partially missing information.
- Missing information has less penalty than incorrect information.

Mark as False ONLY if:
- The system answer contains factually incorrect statements
- There are direct contradictions to known facts
- Information is misleading or wrong

**Do NOT penalize for being incomplete or brief. Only penalize actual errors.**

## 2. Completeness (0-5 scale)
**IMPORTANT**: Evaluate ONLY against the original question/prompt. DO NOT compare to the reference answer!

Score based on:
- 0: Completely off-topic or no answer
- 1-2: Minimal answer, major aspects of the question unanswered
- 3: Partial answer, covers some key points but misses important aspects
- 4: Good answer, addresses most aspects of the question
- 5: Comprehensive answer, thoroughly addresses all aspects of the question

**Reference answer is only for factual checking, NOT for completeness scoring.**

## 3. Extra Information (descriptive text)
Describe any information in the system answer that:
- Is NOT relevant to answering the question
- Adds unnecessary details that don't help answer the question
- Is tangential or off-topic

If the extra information is helpful context or directly relevant to the question, mention that it's relevant.
If there's no extra information or all information is relevant, write "None" or "All information is relevant".

---

**Summary of approach:**
- Factual Correctness: Is anything WRONG? (not: is everything there?)
- Completeness: How well does the answer address the QUESTION? (not: how similar to reference?)
- Extra Information: What's included that doesn't help answer the question?"""


class LLmJudger(BaseJudger):
    def judge(self, judge_input: AnsweredTestCase) -> Optional[JudgementOutput]:
        if judge_input["references"]["answer"] is None:
            return None

        prompt_parts = []

        if judge_input["prompt"]:
            prompt_parts.append(f"**Question:**\n{judge_input['prompt']}\n")

        prompt_parts.append(f"""**Reference Answer (Ground Truth):**
{judge_input["references"]["answer"]}
(Use this ONLY to check factual correctness. Do NOT use it for completeness evaluation.)

**System Answer (To Evaluate):**
{judge_input["system"]["answer"]}

---

**Evaluation Instructions:**

1. **Factual Correctness**: Check if the System Answer contains any factually WRONG information compared to the Reference Answer. If everything stated is correct (even if brief), mark True. Only mark False if there are actual errors or contradictions.

2. **Completeness**: Evaluate how well the System Answer addresses the QUESTION above (NOT how similar it is to the Reference Answer). Score 0-5 based on how thoroughly the question is answered.

3. **Extra Information**: Note any information in the System Answer that is NOT relevant or helpful for answering the Question. If all information is relevant, write "None" or "All information is relevant".

Evaluate the system answer based on the criteria above.""")

        user_content = "\n".join(prompt_parts)

        response = llm.with_structured_output(JudgementOutput).invoke(
            [{"role": "system", "content": JUDGER_SYSTEM_PROMPT}, {"role": "user", "content": user_content}]
        )
        return response

    def aggregate(self, judgements: list[JudgementOutput]) -> Optional[AggregatedJudgement]:
        if not all(isinstance(j, JudgementOutput) for j in judgements):
            return None

        total_completeness = sum(j.completeness for j in judgements)
        total_factual_correctness = sum(1 for j in judgements if j.factual_correctness)
        total_extra_information = sum(
            1 for judgement in judgements if judgement.extra_information.lower() not in ("", "n/a")
        )

        average_completeness = total_completeness / len(judgements)
        average_factual_correctness = total_factual_correctness / len(judgements)
        average_extra_information = total_extra_information / len(judgements)

        return AggregatedJudgement(
            average_factual_correctness=average_factual_correctness,
            average_completeness=average_completeness,
            average_extra_information=average_extra_information,
        )

    def is_compatible(self, judgement: BaseModel) -> bool:
        return isinstance(judgement, JudgementOutput)

    def parse_judgement(self, data: object) -> JudgementOutput | None:
        if isinstance(data, JudgementOutput):
            return data
        if isinstance(data, dict):
            try:
                return JudgementOutput.model_validate(data)
            except Exception:
                return None
        return None

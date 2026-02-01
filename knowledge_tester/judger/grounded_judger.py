from typing import Optional

from judger.base_judger import BaseJudger
from pydantic import BaseModel, Field
from utils.common_types import AnsweredTestCase
from utils.components import llm


class GroundedJudgementOutput(BaseModel):
    hallucination: int = Field(
        ...,
        description="An integer score (0-5) indicating the level of hallucination in the system answer. 0 means no hallucination, 5 means very high hallucination.",
    )
    proof: str = Field(
        ...,
        description="Brief explanation of why the answer is hallucinated, or poorly grounded. If correct and well-grounded, this should be empty or contain 'N/A'.",
    )


class AggregatedGroundedJudgement(BaseModel):
    average_hallucination: float = Field(
        ...,
        description="Average hallucination score across multiple judgements, represented as a float between 0 and 1.",
    )


GROUNDED_JUDGER_SYSTEM_PROMPT = """You are an expert evaluator assessing the correctness and grounding of AI-generated answers against reference answers and source materials.

Your task is to determine if the system's answer is correct and properly grounded by comparing it to:
1. The system chunks (the actual chunks retrieved by the system)
2. The system answer (the final answer provided by the system)

Evaluation criteria:
- **Hallucination**: Does it add information not present in the provided reference materials? Score from 0 (no hallucination) to 5 (severe hallucination).

Guidelines:
- The system answer should primarily use information from the system chunks to be considered well-grounded
- Minor wording differences are acceptable if the meaning is preserved and the source material is correctly used
- If the system answer adds information not in the system chunks, increase the hallucination score
- If the system answer contradicts the system chunks, mark it as incorrect, then increase the hallucination score
- Pay special attention to whether the answer could only be derived from the system chunks vs. general knowledge

When marking an answer hallucinated, provide a clear, specific proof explaining what is wrong, missing, contradictory, or improperly sourced."""


class GroundedJudger(BaseJudger):
    def judge(self, judge_input: AnsweredTestCase) -> Optional[GroundedJudgementOutput]:
        prompt_parts = []

        prompt_parts.append(f"""**System Chunks (Retrieved by System):**
{judge_input["system"]["chunks"]}""")

        prompt_parts.append(f"""**System Answer (To Evaluate):**
{judge_input["system"]["answer"]}""")

        user_content = "\n".join(prompt_parts)
        response = llm.with_structured_output(GroundedJudgementOutput).invoke(
            [{"role": "system", "content": GROUNDED_JUDGER_SYSTEM_PROMPT}, {"role": "user", "content": user_content}]
        )
        return response

    def aggregate(self, judgements: list[GroundedJudgementOutput]) -> Optional[AggregatedGroundedJudgement]:
        if not all(isinstance(j, GroundedJudgementOutput) for j in judgements):
            return None

        total_hallucination = sum(j.hallucination for j in judgements)
        average_hallucination = total_hallucination / len(judgements)

        return AggregatedGroundedJudgement(
            average_hallucination=average_hallucination,
        )

    def is_compatible(self, judgement: BaseModel) -> bool:
        return isinstance(judgement, GroundedJudgementOutput)

    def parse_judgement(self, data: object) -> GroundedJudgementOutput | None:
        if isinstance(data, GroundedJudgementOutput):
            return data
        if isinstance(data, dict):
            try:
                return GroundedJudgementOutput.model_validate(data)
            except Exception:
                return None
        return None

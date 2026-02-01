from typing import Optional

from judger.base_judger import BaseJudger
from pydantic import BaseModel, Field
from utils.common_types import AnsweredTestCase


class KeywordJudgementOutput(BaseModel):
    required_keyword_correctness: float = Field(
        ..., description="Percentage of required keywords that were found in the system answer."
    )
    block_keyword_correctness: float = Field(
        ..., description="Percentage of blocked keywords that were NOT found in the system answer."
    )
    not_found_required_keywords: list[str] = Field(
        default_factory=list, description="List of required keywords that were not found in the system answer."
    )
    blocked_keywords_found: list[str] = Field(
        default_factory=list, description="List of blocked keywords that were found in the system answer."
    )


class AggregatedKeywordJudgement(BaseModel):
    average_required_keyword_correctness: float = Field(
        ..., description="Average required keyword correctness across multiple judgements."
    )
    average_block_keyword_correctness: float = Field(
        ..., description="Average block keyword correctness across multiple judgements."
    )


class KeywordJudger(BaseJudger):
    def judge(self, judge_input: AnsweredTestCase) -> Optional[KeywordJudgementOutput]:
        if not judge_input.get("references") or not judge_input["references"].get("keywords"):
            return None

        required_keywords = judge_input["references"]["keywords"].get("required", [])
        blocked_keywords = judge_input["references"]["keywords"].get("blocked", [])
        system_answer = judge_input["system"]["answer"] if judge_input.get("system") else ""

        not_found_required = [kw for kw in required_keywords if kw.lower() not in system_answer.lower()]
        blocked_found = [kw for kw in blocked_keywords if kw.lower() in system_answer.lower()]

        required_correctness = (
            (len(required_keywords) - len(not_found_required)) / len(required_keywords) if required_keywords else 1.0
        )
        block_correctness = (
            (len(blocked_keywords) - len(blocked_found)) / len(blocked_keywords) if blocked_keywords else 1.0
        )

        return KeywordJudgementOutput(
            required_keyword_correctness=required_correctness,
            block_keyword_correctness=block_correctness,
            not_found_required_keywords=not_found_required,
            blocked_keywords_found=blocked_found,
        )

    def aggregate(self, judgements: list[KeywordJudgementOutput]) -> Optional[AggregatedKeywordJudgement]:
        if not all(isinstance(j, KeywordJudgementOutput) for j in judgements):
            return None

        total_required_correctness = sum(j.required_keyword_correctness for j in judgements)
        total_block_correctness = sum(j.block_keyword_correctness for j in judgements)

        average_required_correctness = total_required_correctness / len(judgements)
        average_block_correctness = total_block_correctness / len(judgements)

        return AggregatedKeywordJudgement(
            average_required_keyword_correctness=average_required_correctness,
            average_block_keyword_correctness=average_block_correctness,
        )

    def is_compatible(self, judgement: BaseModel) -> bool:
        return isinstance(judgement, KeywordJudgementOutput)

    def parse_judgement(self, data: object) -> KeywordJudgementOutput | None:
        if isinstance(data, KeywordJudgementOutput):
            return data
        if isinstance(data, dict):
            try:
                return KeywordJudgementOutput.model_validate(data)
            except Exception:
                return None
        return None

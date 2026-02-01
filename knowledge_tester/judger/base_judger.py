from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel
from utils.common_types import AnsweredTestCase


class BaseJudger(ABC):
    @abstractmethod
    def judge(self, judge_input: AnsweredTestCase) -> Optional[BaseModel]:
        pass

    @abstractmethod
    def aggregate(self, judgements: list[BaseModel]) -> BaseModel:
        pass

    @abstractmethod
    def is_compatible(self, judgement: BaseModel) -> bool:
        pass

    @abstractmethod
    def parse_judgement(self, data: object) -> BaseModel | None:
        """Parse a raw judgement (usually a dict) into the judger's BaseModel.

        Return None if parsing fails or the input is not compatible.
        """
        pass

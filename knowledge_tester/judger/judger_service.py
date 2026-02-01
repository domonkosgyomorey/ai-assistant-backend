from typing import Optional

from judger.base_judger import BaseJudger
from judger.correctness_judger import LLmJudger
from judger.grounded_judger import GroundedJudger
from judger.keyword_judger import KeywordJudger
from pydantic import BaseModel


class JudgerService:
    def __init__(self):
        self.judgers = {}

    def register_judger(self, name: str, judger: BaseJudger):
        self.judgers[name] = judger

    def get_judger(self, name: str) -> BaseJudger:
        return self.judgers.get(name)

    def get_judgers(self) -> dict[str, BaseJudger]:
        return self.judgers

    def list_judgers(self) -> list[str]:
        return list(self.judgers.keys())

    def aggregate(self, judgements: list[BaseModel]) -> Optional[BaseModel]:
        for judger in self.judgers.values():
            parsed = []
            failed = False
            for j in judgements:
                try:
                    parsed_j = judger.parse_judgement(j)
                except Exception:
                    parsed_j = None
                if parsed_j is None:
                    failed = True
                    break
                parsed.append(parsed_j)

            if failed:
                continue

            if all(judger.is_compatible(p) for p in parsed):
                return judger.aggregate(parsed)

        return None


judger_service = JudgerService()
judger_service.register_judger("correctness", LLmJudger())
judger_service.register_judger("keyword", KeywordJudger())
judger_service.register_judger("grounded", GroundedJudger())

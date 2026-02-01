import argparse
import json
from collections import defaultdict
from pathlib import Path

from judger.judger_service import judger_service
from pydantic import BaseModel
from utils.common_types import JudgedTestCase
from utils.safe_utils import safe_open

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate metrics from knowledge tester results.")
    parser.add_argument(
        "--current",
        type=str,
        required=True,
        help="Path to the judged test case JSON file.",
    )
    args = parser.parse_args()
    current_file_path = Path(args.current)

    with safe_open(current_file_path, mode="r", encoding="utf-8") as f:
        current_data: list[JudgedTestCase] = [JudgedTestCase(**case) for case in json.load(f)]

    group_by_group: dict[str, list[JudgedTestCase]] = defaultdict(list)
    for case in current_data:
        group = case.groups
        if not group:
            group = ["default"]
        group_by_group[group[-1]].append(case)

    group_by_question_type: dict[str, list[JudgedTestCase]] = defaultdict(list)
    for case in current_data:
        question_type = case.question_type
        group_by_question_type[question_type].append(case)

    group_by_group_result: dict[str, dict[str, BaseModel]] = {}
    for group, cases in group_by_group.items():
        group_by_judgement: dict[str, list[BaseModel]] = {}
        for case in cases:
            for name, metrics in case.judgement.items():
                if name not in group_by_judgement:
                    group_by_judgement[name] = []
                group_by_judgement[name].append(metrics)

        for name, metrics_list in group_by_judgement.items():
            aggregated_metrics = judger_service.aggregate(metrics_list)
            if group not in group_by_group_result:
                group_by_group_result[group] = {}
            group_by_group_result[group][name] = aggregated_metrics

    group_by_question_type_result: dict[str, dict[str, BaseModel]] = {}
    for question_type, cases in group_by_question_type.items():
        group_by_judgement: dict[str, list[BaseModel]] = {}
        for case in cases:
            for name, metrics in case.judgement.items():
                if name not in group_by_judgement:
                    group_by_judgement[name] = []
                group_by_judgement[name].append(metrics)

        for name, metrics_list in group_by_judgement.items():
            aggregated_metrics = judger_service.aggregate(metrics_list)
            if question_type not in group_by_question_type_result:
                group_by_question_type_result[question_type] = {}
            group_by_question_type_result[question_type][name] = aggregated_metrics

    print("Metrics by Group:")
    print(json.dumps(group_by_group_result, indent=2, default=str))
    print("\nMetrics by Question Type:")
    print(json.dumps(group_by_question_type_result, indent=2, default=str))

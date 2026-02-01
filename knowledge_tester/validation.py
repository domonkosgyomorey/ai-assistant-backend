import json
import os
import sys

from pydantic import ValidationError
from utils.common_types import TestCase
from utils.safe_utils import safe_open, safe_walk

# Collect all validation errors
errors = []


def validate_test_case(file_path: str):
    try:
        with safe_open(file_path, mode="r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        errors.append(f"Error decoding JSON in file: {file_path}")
        return
    except Exception as e:
        errors.append(f"Error reading file {file_path}: {e}")
        return

    if not isinstance(data, list):
        errors.append(f"Test case file {file_path} does not contain a list of test cases.")
        return

    for item in data:
        try:
            TestCase.model_validate(item)
        except ValidationError as e:
            errors.append(f"Validation failed for {file_path}: {e}")


def validate_folder(folder_path: str):
    for root, _, files in safe_walk(folder_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, folder_path)
                validate_test_case(os.path.join(folder_path, rel_path))


if __name__ == "__main__":
    validate_folder(os.path.join("knowledge_tester", "test_cases"))

    if errors:
        print("Validation errors detected:\n")
        for err in errors:
            print(err)
        sys.exit(1)
    else:
        print("All test cases are valid!")

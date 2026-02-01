import json
import os
import sys
import time
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed

import joblib
from chunk_gatherer.chunk_gatherer_orchestrator import ChunkGathererOrchestrator
from chunk_gatherer.db_gatherer import DBGatherer
from chunk_gatherer.web_gatherer import WebGatherer
from colorama import Fore, Style, init
from judger.judger_service import judger_service
from system_interaction.interaction import Interaction, InteractionMessage
from system_interaction.local_interaction import LocalInteraction
from utils.common_types import JudgedTestCase, TestCase, TestCaseWithGroups
from utils.safe_utils import safe_open, safe_walk
from validation import validate_folder, validate_test_case

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "server")))
from core.config.config import Config

init(autoreset=True)

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "server")))

cache_dir = os.path.join(os.getcwd(), "cache")


def get_cached_progress(cache_key: str) -> dict | None:
    if os.path.exists(cache_key):
        return joblib.load(cache_key)
    return None


def progress_caching(cache_key: str, obj: dict):
    joblib.dump(obj, cache_key)


def json_cache_load(file_path: str) -> dict | None:
    if os.path.exists(file_path):
        with safe_open(file_path, mode="r", encoding="utf-8") as f:
            return json.load(f)
    return None


def json_cache_save(file_path: str, obj: dict):
    with safe_open(file_path, mode="w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def fetch_chunks_for_source(chunk_gatherer: ChunkGathererOrchestrator, source: str, source_type: str) -> list[str]:
    """Fetch chunks based on source type."""
    if source_type == "file":
        chunk = chunk_gatherer.db_gatherer.gather(source)
    elif source_type == "web":
        chunk = chunk_gatherer.web_gatherer.gather(source)
    else:
        chunk = chunk_gatherer.gather_chunks(key=source)
    return [chunk] if chunk else []


def gather_system_chunks(
    chunk_gatherer: ChunkGathererOrchestrator, system_answer: str, system_sources: dict
) -> tuple[str, float]:
    """Gather all relevant chunks from system sources in parallel.

    Args:
        chunk_gatherer: Orchestrator for gathering chunks
        system_answer: The system's answer text
        system_sources: Dict mapping source identifiers to their types ("file" or "web")

    Returns:
        Tuple of (joined chunks string, elapsed time in seconds)
    """
    relevant_chunks = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(fetch_chunks_for_source, chunk_gatherer, source, source_type): source
            for source, source_type in system_sources.items()
        }

        for future in as_completed(futures):
            try:
                chunks = future.result()
                relevant_chunks.extend(chunks)
            except Exception as e:
                source = futures[future]
                print(f"\n{Fore.RED}✗ Error fetching chunks for source '{source}': {e}{Style.RESET_ALL}")

    elapsed_time = time.time() - start_time
    return "\n".join(relevant_chunks), elapsed_time


def judge_single_case(case: TestCase, idx: int) -> tuple[JudgedTestCase, bool, float]:
    """
    Judge a single test case with chunk gathering and evaluation.

    Args:
        case: The test case to judge
        idx: Index of the test case (for error reporting)

    Returns:
        Tuple of (updated test case, success flag)
    """
    try:
        case["judgement"] = {}
        for name, judger in judger_service.get_judgers().items():
            judgement = judger.judge(case)
            if judgement is None:
                continue
            case["judgement"][name] = judgement.model_dump()

        return case, True

    except Exception as e:
        print(f"\n{Fore.RED}✗ Error judging test case {idx}: {e}{Style.RESET_ALL}")
        return case, False, 0.0


def read_input(input_path: str) -> list[TestCaseWithGroups]:
    """Read input test cases from a JSON file or directory."""
    test_cases: list[TestCaseWithGroups] = []
    if os.path.isdir(input_path):
        validate_folder(input_path)
        root_dir = None
        for dir_path, _, file_names in safe_walk(input_path):
            if root_dir is None:
                root_dir = dir_path
            rel_dir = os.path.relpath(dir_path, root_dir)
            names = [] if rel_dir == "." else rel_dir.split(os.sep)

            for file in file_names:
                if file.endswith(".json"):
                    file_path = os.path.join(dir_path, file)
                    rel_path = os.path.relpath(file_path, input_path)
                    validate_test_case(os.path.join(input_path, rel_path))
                    with safe_open(file_path, mode="r", encoding="utf-8") as f:
                        test_cases_part: list[TestCaseWithGroups] = json.load(f)
                        for case in test_cases_part:
                            case["groups"] = names
                        test_cases.extend(test_cases_part)
                    print(f"{Fore.GREEN}✓ Loaded {len(test_cases)} test cases from {rel_path}\n")
    else:
        validate_test_case(input_path)
        with safe_open(input_path, mode="r", encoding="utf-8") as f:
            test_cases_part = json.load(f)
            for case in test_cases_part:
                case["groups"] = []
            test_cases.extend(test_cases_part)
        print(f"{Fore.GREEN}✓ Loaded {len(test_cases)} test cases from {input_path}\n")
    return test_cases


def answering(
    interaction: Interaction, chunk_gatherer: ChunkGathererOrchestrator, test_cases: list[TestCaseWithGroups]
):
    print(f"{Fore.BLUE}🚀 Starting fresh evaluation...\n")
    print(f"{Fore.CYAN}{'─' * 80}")
    print(f"{Fore.CYAN}Phase 1: Generating System Answers")
    print(f"{Fore.CYAN}{'─' * 80}{Style.RESET_ALL}\n")

    def evaluate_case(case: TestCase):
        response = interaction.send_message(InteractionMessage(prompt=case["prompt"]))
        case["system"] = {}
        case["system"]["answer"] = response["response"]
        groundings = response["groundings"]
        case["system"]["sources"] = {v["source"]: v["type"] for v in groundings}
        chunks_result, _ = gather_system_chunks(
            chunk_gatherer=chunk_gatherer,
            system_answer=case["system"]["answer"],
            system_sources=case["system"]["sources"],
        )
        case["system"]["chunks"] = chunks_result

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(evaluate_case, case) for case in test_cases]
        for future in as_completed(futures):
            future.result()

    print(f"\n{Fore.GREEN}✓ All system answers generated successfully!")
    print(f"{Fore.BLUE}💾 Saving progress to 'answered_test_cases.json'...\n")
    return test_cases


def judging(test_cases: list[JudgedTestCase]):
    print(f"{Fore.CYAN}{'─' * 80}")
    print(f"{Fore.CYAN}Phase 2: Judging System Answers")
    print(f"{Fore.CYAN}{'─' * 80}{Style.RESET_ALL}\n")

    judged_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(judge_single_case, case, idx): idx for idx, case in enumerate(test_cases, 1)}

        for future in as_completed(futures):
            idx = futures[future]
            result_case, success = future.result()
            test_cases[idx - 1] = result_case

            if success:
                judged_count += 1
            else:
                error_count += 1

    print(f"\n{Fore.GREEN}✓ Judgement completed!")
    print(f"{Fore.GREEN}  • Successfully judged: {judged_count}/{len(test_cases)}")
    if error_count > 0:
        print(f"{Fore.RED}  • Errors encountered: {error_count}")
    print()


def register_judgers():
    print(f"{Fore.YELLOW}📋 Registered Judgers:")
    for judger_name in judger_service.get_judgers().keys():
        print(f"   • {judger_name}")
    print()


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate knowledge tester results.")
    parser.add_argument(
        "--input", "-i", required=True, help="Path to the JSON file or directory containing test cases."
    )
    parser.add_argument("--cache-answer", "-c", action="store_true", help="Enable caching of system answers.")

    args = parser.parse_args()
    input_path = args.input
    cache_answers = args.cache_answer

    print(f"{Fore.CYAN}{'=' * 80}")
    print(f"{Fore.CYAN}🧪 Knowledge Tester Evaluation Pipeline")
    print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}\n")

    test_cases = read_input(input_path)

    interaction = LocalInteraction()
    chunk_gatherer = ChunkGathererOrchestrator(
        db_gatherer=DBGatherer(collection_name=Config.mongo.COLLECTION_NAME),
        web_gatherer=WebGatherer(
            api_key=Config.tavily.API_KEY,
            delay_between_requests=1.5,  # 1.5 seconds between requests
            max_retries=3,  # Retry up to 3 times with exponential backoff
        ),
    )

    print(f"{Fore.YELLOW}📋 Registered Judgers:")
    for judger_name in judger_service.get_judgers().keys():
        print(f"   • {judger_name}")
    print()

    cached_progress = json_cache_load("answered_test_cases.json") if cache_answers else None

    if cached_progress is not None:
        print(f"{Fore.YELLOW}💾 Loaded cached progress with {len(cached_progress)} answered test cases.")
        test_cases = cached_progress
    else:
        test_cases = answering(interaction, chunk_gatherer, test_cases)

    if cache_answers:
        json_cache_save("answered_test_cases.json", test_cases)

    judging(test_cases)

    print(f"{Fore.BLUE}💾 Saving final results to 'judged_test_cases.json'...")
    json_cache_save("judged_test_cases.json", test_cases)

    print(f"\n{Fore.CYAN}{'=' * 80}")
    print(f"{Fore.GREEN}✅ Evaluation completed successfully!")
    print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}\n")

    chunk_gatherer.db_gatherer.close()

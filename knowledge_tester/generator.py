import argparse
import json
import os
import random
from typing import TypedDict

from processor import process_documents
from pydantic import BaseModel, Field
from tqdm import tqdm
from utils.common_types import Cluster, CustomDocument
from utils.components import llm
from utils.prompts import GENERATOR_LLM_SYSTEM_PROMPT, GENERATOR_USER_PROMPT, QuestionType, get_prompt
from utils.safe_utils import safe_open


class LLMOutputStructure(BaseModel):
    question: str = Field(..., description="The generated user question, thats short and natural.")
    answer: str = Field(
        ..., description="The accurate and shorter reference answer for the question from the provided content."
    )


class QuestionTypeRatio(TypedDict):
    question_type: QuestionType
    ratio: float


def generate_test_case(
    context: str, previous_questions: list[str], question_type: QuestionType, keywords: str, examples: list[str]
) -> LLMOutputStructure:
    prev_questions = ", ".join(previous_questions)
    return llm.with_structured_output(LLMOutputStructure).invoke(
        [
            {
                "role": "system",
                "content": GENERATOR_LLM_SYSTEM_PROMPT.format(
                    previous_questions=prev_questions,
                    examples=examples,
                    question_type=get_prompt(question_type),
                ),
            },
            {"role": "user", "content": GENERATOR_USER_PROMPT.format(combined_content=context, keywords=keywords)},
        ]
    )


def select_up_to_n_len(context: list[CustomDocument], n: int) -> list[CustomDocument]:
    result = []
    all_len = 0
    sampled_docs = random.sample(context, min(len(context), 25))
    for doc in sampled_docs:
        if all_len + len(doc["content"]) > n:
            break
        all_len += len(doc["content"])
        result.append(doc)

    return result


def generator(
    clusters: list[Cluster],
    number_of_test_cases: int | list[int],
    max_context_len_per_question: int,
    question_type_ratios: list[QuestionTypeRatio],
    examples: list[str],
) -> list[dict]:
    # Group clusters by file
    clusters_by_file: dict[str, list[Cluster]] = {}
    for cluster in clusters:
        source_file = cluster["source"]
        clusters_by_file.setdefault(source_file, []).append(cluster)

    num_files = len(clusters_by_file)

    if isinstance(number_of_test_cases, list) and len(number_of_test_cases) == num_files:
        file_test_cases = dict(zip(clusters_by_file.keys(), number_of_test_cases, strict=False))
        total_cases = sum(number_of_test_cases)
    else:
        cases_per_file = number_of_test_cases // num_files
        extra_cases = number_of_test_cases % num_files

        file_test_cases = {}
        for i, file_path in enumerate(clusters_by_file.keys()):
            file_test_cases[file_path] = cases_per_file + (1 if i < extra_cases else 0)
        total_cases = number_of_test_cases

    print(f"\n� Test Case Distribution ({total_cases} total):")
    print("─" * 80)

    max_filename_len = max(len(os.path.basename(f)) for f in file_test_cases.keys())

    for file_path, cases_count in file_test_cases.items():
        filename = os.path.basename(file_path)

        type_counts = {}
        for qtr in question_type_ratios:
            type_counts[qtr["question_type"].value] = int(qtr["ratio"] * cases_count)

        # Adjust for rounding errors
        calculated_total = sum(type_counts.values())
        if calculated_total < cases_count:
            for qtr in question_type_ratios:
                if qtr["ratio"] > 0:
                    type_counts[qtr["question_type"].value] += cases_count - calculated_total
                    break

        type_display = ", ".join([f"{type_name}: {count:>2}" for type_name, count in type_counts.items()])
        print(f"{filename:<{max_filename_len + 2}} {type_display}, total: {cases_count:>2}")

    print("─" * 80)

    # Flatten all clusters from all files into a single list
    all_clusters = []
    for file_path in file_test_cases.keys():
        all_clusters.extend(clusters_by_file[file_path])

    if not all_clusters:
        print("Error: No clusters found!")
        return []

    total_test_cases = sum(file_test_cases.values())

    # Prepare question types with proper ratios
    question_types = []
    for ratio in question_type_ratios:
        question_types.extend([ratio["question_type"]] * int(ratio["ratio"] * total_test_cases))
    question_types.extend([question_type_ratios[0]["question_type"]] * (total_test_cases - len(question_types)))
    random.shuffle(question_types)

    # Sequential round-robin generation with global previous_questions tracking
    test_cases = []
    previous_questions = []  # Global list shared across all clusters

    print("\n🔄 Generating test cases in round-robin order...")

    with tqdm(total=total_test_cases, desc="Generating test cases") as pbar:
        for i in range(total_test_cases):
            # Round-robin: cycle through clusters
            cluster_idx = i % len(all_clusters)
            cluster = all_clusters[cluster_idx]
            question_type = question_types[i]

            # Select random chunks from cluster
            chunks = select_up_to_n_len(cluster["documents"], max_context_len_per_question)
            combined_content = "\n\n".join(
                [f"**Source**: {doc['file_name']}\n**Content:**{doc['content']}" for doc in chunks]
            )

            # Generate test case with ALL previous questions (from all clusters)
            structure = generate_test_case(
                combined_content,
                previous_questions.copy(),  # Pass all previous questions
                question_type,
                cluster["description"],
                examples,
            )

            sources = {}
            for doc in chunks:
                key = doc["file_name"]
                if doc["page_number"]:
                    key += "#page=" + str(doc["page_number"])
                sources[key] = doc["source_type"]

            test_case = {
                "prompt": structure.question,
                "question_type": str(question_type),
                "references": {
                    "answer": structure.answer,
                    "chunks": combined_content,
                    "sources": sources,
                },
                "generation_metadata": {
                    "cluster_description": cluster["description"],
                },
            }

            test_cases.append(test_case)
            previous_questions.append(structure.question)  # Add to global list
            pbar.update(1)

    return test_cases


def read_examples(file_path: str) -> list[str]:
    if file_path is None:
        return []
    examples = []
    with safe_open(file_path, mode="r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(line)
    return examples


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--context-path",
        "-c",
        type=str,
        nargs="+",
        required=True,
        help="One or more paths to context documents (PDF, TXT, or DOCX files)",
    )
    arg_parser.add_argument(
        "--num-test-cases",
        "-n",
        type=int,
        nargs="+",
        default=[5],
        help="Number of test cases. Single number = distribute evenly. Multiple numbers = per-file counts (must match file count)",
    )
    arg_parser.add_argument("--output-path", "-o", type=str, required=True)
    arg_parser.add_argument("--max-context-len-per-question", "-m", type=int, default=8000)
    arg_parser.add_argument("--ignored-keywords", "-i", type=str, nargs="*", default=[])
    for question_type in QuestionType:
        arg_name = f"--{question_type.value.replace('_', '-')}-ratio"
        arg_parser.add_argument(
            arg_name, type=float, required=True, help=f"Ratio for {question_type.value.upper()} question type"
        )
    arg_parser.add_argument(
        "--separate-clustering",
        "-s",
        action="store_true",
        help="Cluster each file separately instead of combining all documents",
    )
    arg_parser.add_argument(
        "--examples", required=False, default=None, help="Path to a txt file that contains the examples"
    )
    args = arg_parser.parse_args()

    context_paths = args.context_path
    output_path = args.output_path
    num_test_cases = args.num_test_cases
    max_context_len_per_question = args.max_context_len_per_question
    ignored_keywords = args.ignored_keywords
    separate_clustering = args.separate_clustering
    examples_path = args.examples

    if examples_path is not None and (
        not os.path.exists(examples_path) or examples_path.split(".")[-1].lower() != "txt"
    ):
        print(f"Examples file not found at: {examples_path}")
        exit(1)

    question_type_ratios: list[QuestionTypeRatio] = []
    total_ratio = 0.0

    for question_type in QuestionType:
        ratio_attr = f"{question_type.value.replace('_', '-')}_ratio"
        ratio_value = getattr(args, ratio_attr.replace("-", "_"))
        question_type_ratios.append({"question_type": question_type, "ratio": ratio_value})
        total_ratio += ratio_value

    if abs(total_ratio - 1.0) > 0.01:
        print(f"Error: The sum of question type ratios must equal 1.0, but got {total_ratio:.3f}")
        ratio_details = ", ".join([f"{qtr['question_type'].value}: {qtr['ratio']}" for qtr in question_type_ratios])
        print(f"Ratios: {ratio_details}")
        exit(1)

    print("Processing documents and creating clusters...")
    clustered_chunks = process_documents(context_paths, ignored_keywords, separate_clustering)

    if len(num_test_cases) == 1:
        total_test_cases = num_test_cases[0]
        print(f"\nGenerating {total_test_cases} test cases (distributed evenly)...")
    else:
        num_files = len(set(cluster["source"] for cluster in clustered_chunks))
        if len(num_test_cases) != num_files:
            print(f"Error: You provided {len(num_test_cases)} test case counts, but there are {num_files} files.")
            print(f"Either provide 1 number (even distribution) or {num_files} numbers (per-file counts).")
            exit(1)
        total_test_cases = sum(num_test_cases)
        print(f"\nGenerating {num_test_cases} test cases per file (total: {total_test_cases})...")

    test_cases = generator(
        clustered_chunks,
        num_test_cases,
        max_context_len_per_question=max_context_len_per_question,
        question_type_ratios=question_type_ratios,
        examples=read_examples(examples_path),
    )

    with safe_open(output_path, mode="w", encoding="utf-8") as f:
        f.write(json.dumps(test_cases, indent=4, ensure_ascii=False))

    print(f"\nGenerated {len(test_cases)} test cases")
    print(f"Test cases saved to: {output_path}")

from argparse import ArgumentParser

from evaluation.utils.giskard_helper import GiskardHelper

if __name__ == "__main__":
    parser = ArgumentParser(description="Generate testset using Giskard")
    parser.add_argument(
        "--load-cached-knowledge", "-k", action="store_true", default=False, help="Load knowledge base from cache."
    )
    parser.add_argument("--number-of-samples", "-n", type=int, default=5, help="Number of samples to generate.")
    parser.add_argument("--upload-results", "-u", action="store_true", default=False, help="Upload results to GCP.")

    args = parser.parse_args()
    load_cached_knowledge = args.load_cached_knowledge
    number_of_samples = args.number_of_samples
    upload_results = args.upload_results

    giskard_helper = GiskardHelper(load_cached_knowledge)
    giskard_helper.testset_generation(
        number_of_samples=number_of_samples,
        agent_description="This agent can answer questions about University of Obuda in topics such as students' requirements, administration and etc.",
        output_path="testset.jsonl",
        upload_results=upload_results,
    )

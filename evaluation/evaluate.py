from argparse import ArgumentParser

from evaluation.utils.giskard_helper import GiskardHelper

if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate testset using Giskard")
    parser.add_argument(
        "--load-cached-knowledge", "-k", action="store_true", default=False, help="Load knowledge base from cache."
    )
    parser.add_argument("--upload-results", "-u", action="store_true", default=False, help="Upload results to GCP.")
    args = parser.parse_args()
    load_cached_knowledge = args.load_cached_knowledge
    upload_results = args.upload_results

    giskard_helper = GiskardHelper(load_knowledge_from_cache=load_cached_knowledge)
    testset = giskard_helper.testset_evaluation(upload_results=upload_results)
    print("Testset evaluation completed.")

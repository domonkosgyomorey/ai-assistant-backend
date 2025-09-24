from evaluation.giskard_helper import GiskardHelper

if __name__ == "__main__":
    giskard_helper = GiskardHelper(load_knowledge_from_cache=True)
    testset = giskard_helper.testset_evaluation(upload_results=True)
    print("Testset evaluation completed.")

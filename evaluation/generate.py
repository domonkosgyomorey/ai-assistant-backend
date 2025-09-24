from evaluation.giskard_helper import GiskardHelper

if __name__ == "__main__":
    giskard_helper = GiskardHelper()
    giskard_helper.testset_generation(
        number_of_samples=5,
        agent_description="This agent can answer questions about University of Obuda in topics such as students' requirements, administration and etc.",
        output_path="testset.jsonl",
        upload_results=True,
    )

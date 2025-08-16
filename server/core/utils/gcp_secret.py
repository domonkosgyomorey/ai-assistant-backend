from google.auth import default
from google.cloud.secretmanager import SecretManagerServiceClient


def get_secret(secret_name: str) -> str:
    """
    Retrieves a secret from Google Cloud Secret Manager.

    Args:
        secret_name (str): The name of the secret to retrieve.

    Returns:
        str: The value of the secret.
    """
    client: SecretManagerServiceClient = SecretManagerServiceClient()
    name = f"projects/{get_active_gcloud_project()}/secrets/{secret_name}/versions/latest"
    response = client.access_secret_version(name=name)
    return response.payload.data.decode("UTF-8")


def get_active_gcloud_project():
    try:
        credentials, project_id = default()
        if project_id:
            return project_id
        else:
            print("No active project found in the environment.")
            return None
    except Exception as e:
        print(f"Error retrieving active project: {e}")
        return None

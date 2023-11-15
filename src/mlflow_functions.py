import mlflow
import util
from mlflow import MlflowClient

config = util.get_config()


def configure_client():
    # Configure MLflow Tracking Client
    client = MlflowClient(tracking_uri="http://127.0.0.1:5000")
    # Use the fluent API to set the tracking uri and the active experiment
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    return client


def create_experiment(client):
    # Create a new MLflow experiment if it doesn't exist
    experiment_tags = {
        "mlflow.note.content": config["experiment_description"],
    }
    if (
        client.search_experiments(
            filter_string=f"name = '{config['experiment_name']}'"
        )
        == []
    ):
        client.create_experiment(
            name=config["experiment_name"], tags=experiment_tags
        )
    else:
        print("Experiment name already exists")

    # Sets the current active experiment
    mlflow.set_experiment(config["experiment_name"])


def log(params, metrics):
    # Define a run name for this iteration of training.
    # If this is not set, a unique name will be auto-generated for your run.
    run_name = None  # config['run_name']

    # Define an artifact path that the model will be saved to.
    # artifact_path = None #config['artifact_path']

    # Initiate the MLflow run context
    with mlflow.start_run(run_name=run_name):
        # Log the parameters used for the model fit
        mlflow.log_params(params)

        # Log the error metrics that were calculated during validation
        mlflow.log_metrics(metrics)

        mlflow.log_artifact("../data/evaluation_results.csv")
        mlflow.log_artifact("../conf/config.yaml")


if __name__ == "__main__":
    client = configure_client()
    create_experiment(client)

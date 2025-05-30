import subprocess
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("python-dotenv not installed — skipping .env loading.")


def configure_mlflow():
    tracking_uri = (
        "https://dagshub.com/ashish.student2025/AirTravel_Sentiment_Analysis.mlflow"
    )
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri

    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")

    if not username or not password:
        print("MLflow credentials are missing.")
    else:
        print(f"MLflow tracking URI set to {tracking_uri}")


def configure_dvc():
    if not Path(".dvc").exists():
        subprocess.run(["dvc", "init"], check=True)
        print("DVC initialized.")
    else:
        print("DVC already initialized.")

    gcp_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not gcp_creds or not Path(gcp_creds).exists():
        print("GCP credentials not found or invalid.")
    else:
        print(f"Using GCP credentials at: {gcp_creds}")

    # Set GCS remote (idempotent)
    subprocess.run(
        [
            "dvc",
            "remote",
            "add",
            "-f",
            "-d",
            "remote",
            "gs://dvc-storage-airtravel-sentiment-analysis/storage",
        ],
        check=True,
    )
    print("DVC GCS remote configured.")


def run_dvc_repro():
    subprocess.run(["dvc", "repro"], check=True)
    print("DVC pipeline reproduced.")


# Auto-configuration block
if __name__ == "__main__":
    configure_mlflow()
    configure_dvc()
    run_dvc_repro()

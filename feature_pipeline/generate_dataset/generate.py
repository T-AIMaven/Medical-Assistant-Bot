import sys
from pathlib import Path
import pandas as pd
import json
import logging
from comet_ml import Artifact, start
from sklearn.model_selection import train_test_split

# To mimic using multiple Python modules, such as 'core' and 'feature_pipeline',
# we will add the './src' directory to the PYTHONPATH. This is not intended for
# production use cases but for development and educational purposes.
ROOT_DIR = str(Path(__file__).parent.parent.parent)
sys.path.append(ROOT_DIR)

from core import get_logger
from core.config import settings

logger = get_logger(__name__)

settings.patch_localhost()
logger.warning(
    "Patched settings to work with 'localhost' URLs. \
    Remove the 'settings.patch_localhost()' call from above when deploying or running inside Docker."
)

logger = get_logger(__name__)

class DatasetUploader:
    def validate_and_upload_csv(self, csv_path: str, data_type: str) -> None:
        try:
            # Read the CSV file
            df = pd.read_csv(csv_path)

            # Validate the CSV file (example: check if required columns are present)
            required_columns = ['question', 'answer']
            if not all(column in df.columns for column in required_columns):
                raise ValueError(f"CSV file must contain the following columns: {required_columns}")

            # Log the validation success
            logger.info(f"CSV file {csv_path} validated successfully.")

            # Prepare the dataset for upload
            generated_instruct_dataset = []
            for _, row in df.iterrows():
                question = row['question']
                answer = row['answer']
                if pd.notna(question) and pd.notna(answer):
                    generated_instruct_dataset.append({"question": question, "answer": answer})

            # Split the dataset into train and test sets
            train_data, test_data = self._split_dataset(generated_instruct_dataset)

            # Upload the dataset to Comet
            self.push_to_comet((train_data, test_data), data_type)

        except Exception as e:
            logger.exception(f"Failed to validate and upload CSV file {csv_path}: {e}")


    def _split_dataset(
        self, generated_instruct_dataset: list[dict], test_size: float = 0.1
    ) -> tuple[list[dict], list[dict]]:
        """Split dataset into train and test sets.

        Args:
            generated_instruct_dataset (dict): Dataset containing content and instruction pairs

        Returns:
            tuple[dict, dict]: Train and test splits of the dataset
        """

        if len(generated_instruct_dataset) == 0:
            return [], []

        train_data, test_data = train_test_split(
            generated_instruct_dataset, test_size=test_size, random_state=42
        )

        return train_data, test_data

    def push_to_comet(
        self,
        train_test_split: tuple[list[dict], list[dict]],
        data_type: str,
        collection_name: str,
        output_dir: Path = Path("generated_dataset"),
    ) -> None:
        output_dir.mkdir(exist_ok=True)

        try:
            logger.info(f"Starting to push data to Comet: {collection_name}")

            experiment = start()

            training_data, testing_data = train_test_split

            file_name_training_data = output_dir / f"{collection_name}_training.json"
            file_name_testing_data = output_dir / f"{collection_name}_testing.json"

            logging.info(f"Writing training data to file: {file_name_training_data}")
            with file_name_training_data.open("w") as f:
                json.dump(training_data, f)

            logging.info(f"Writing testing data to file: {file_name_testing_data}")
            with file_name_testing_data.open("w") as f:
                json.dump(testing_data, f)

            logger.info("Data written to file successfully")

            artifact = Artifact(f"{data_type}-instruct-dataset")
            artifact.add(file_name_training_data)
            artifact.add(file_name_testing_data)
            logger.info(f"Artifact created.")

            experiment.log_artifact(artifact)
            experiment.end()
            logger.info("Artifact pushed to Comet successfully.")

        except Exception:
            logger.exception(
                f"Failed to create Comet artifact and push it to Comet.",
            )

if __name__ == "__main__":
    settings.patch_localhost()
    logger.warning(
        "Patched settings to work with 'localhost' URLs. \
        Remove the 'settings.patch_localhost()' call from above when deploying or running inside Docker."
    )

    csv_path = "dataset/mle_screening_dataset.csv"
    data_type = "csv"

    dataset_uploader = DatasetUploader()
    dataset_uploader.validate_and_upload_csv(csv_path, data_type)
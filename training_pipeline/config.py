from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv()

ROOT_DIR = str(Path(__file__).parent.parent.parent.parent)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR, env_file_encoding="utf-8")

    # Hugging Face config
    HUGGINGFACE_BASE_MODEL_ID: str = "meta-llama/Llama-3.1-8B"
    HUGGINGFACE_ACCESS_TOKEN: str | None = None

    # Comet config
    COMET_API_KEY: str | None = None
    COMET_WORKSPACE: str | None = None
    COMET_PROJECT: str = os.getenv("COMET_PROJECT")

    DATASET_ID: str = "articles-instruct-dataset"  # Comet artifact containing your fine-tuning dataset (available after generating the instruct dataset).


settings = Settings()

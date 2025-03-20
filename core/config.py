from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv()

ROOT_DIR = str(Path(__file__).parent.parent.parent)


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR, env_file_encoding="utf-8")

    # CometML config
    COMET_API_KEY: str | None = os.getenv("COMET_API_KEY")
    COMET_WORKSPACE: str | None = os.getenv("COMET_WORKSPACE")
    COMET_PROJECT: str = os.getenv("COMET_PROJECT")

    # LLM Model config
    HUGGINGFACE_ACCESS_TOKEN: str | None = None
   
settings = AppSettings()


import os
from pathlib import Path

from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()

year = 2017
local_path = Path("/home/jararias/.local/share/pysparta/merra2_daily")

# Sube toda la carpeta local al repo de Hugging Face
api = HfApi()
api.upload_folder(
    repo_id="josearuizarias/merra2-daily-clearsky",
    folder_path=local_path / f"{year}",
    path_in_repo=f"{year}",
    repo_type="dataset",
    run_as_future=True,
    token=os.getenv("HUGGING_FACE_HUB_TOKEN"),
)
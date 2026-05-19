
import os
from pathlib import Path

import huggingface_hub as Hf
from dotenv import load_dotenv

load_dotenv()

LOCAL_REPO_PATH = Path("/home/jararias/.local/share/spartasolar/merra2-daily-for-huggingface")

def delete_Hf_yearly_chunk(year: int):
    Hf.delete_folder(
        repo_id="josearuizarias/merra2-daily-clearsky",
        repo_type="dataset",
        path_in_repo=str(year),
        token=os.getenv("HUGGING_FACE_HUB_TOKEN"))

def upload_Hf_yearly_chunk(year: int):
    Hf.upload_folder(
        repo_id="josearuizarias/merra2-daily-clearsky",
        repo_type="dataset",
        folder_path=LOCAL_REPO_PATH / f"{year}",
        path_in_repo=f"{year}",
        run_as_future=True,
        token=os.getenv("HUGGING_FACE_HUB_TOKEN"))

if __name__ == "__main__":

    for year in range(2015, 1998, -1):
        print(f"Uploading {year}...")
        upload_Hf_yearly_chunk(year)


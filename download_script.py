#!/usr/bin/env python3
"""
Script to download and unzip files from given URLs.

This script defines a function `download_and_unzip_if_not_exists` that downloads
and unzips files from the provided URLs to a specified folder, skipping the process
if the file already exists in the folder.

Usage:
    Modify the `urls` list with the desired URLs.
    Run the script to download and unzip the files.

Author:
    J.A. Busker
"""
import sys
import os
import urllib.request
import shutil
import gzip
from pathlib import Path


URLS: list[str] = [
    "https://dcc.icgc.org/api/v1/download?fn=/PCAWG/mutational_signatures/Input_Data_PCAWG7_23K_Spectra_DB/vcf_like_simple_files/WES_Other.20180327.simple.gz",
    "https://dcc.icgc.org/api/v1/download?fn=/PCAWG/mutational_signatures/Input_Data_PCAWG7_23K_Spectra_DB/vcf_like_simple_files/WGS_Other.20180413.simple.gz",
]


# Function to download and unzip files
def download_and_unzip(url: str, folder: Path) -> None:
    """
    Download and unzip a file from the given URL to the specified folder if it doesn't already exist.

    Args:
        url (str): The URL of the file to be downloaded.
        folder (Path, optional): The folder where the file should be saved. Defaults to the current directory.

    Returns:
        None
    """
    # Extract the file name from the URL
    output_file: str = ".".join(Path(url).parts[-1].split(".")[:-1])
    target_file: Path = folder / output_file

    # Check if the file already exists
    if target_file.exists():
        print(f"File {output_file} already exists. Skipping download.")
        return
    # Download the file
    download_path = folder / f"{output_file}.gz"
    print(f"Downloading {url} to {download_path}...")
    urllib.request.urlretrieve(url, download_path)
    # Unzip the file
    print(f"Unzipping {output_file}.gz to {folder}...")
    with gzip.open(download_path, "rb") as f_in:
        with open(folder / output_file, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    # Remove the downloaded compressed fil
    print("Cleaning up...")
    os.remove(download_path)
    print(f"File {output_file} is done!")


def main() -> int:
    """
    Main function to execute the download and unzip process.

    Returns:
        int: Exit code (0 for success).
    """
    # Get the path of the current Python script
    script_path: Path = Path(__file__).parent
    # Create the data folder if it doesn't exist
    data_folder: Path = script_path / "data" / "vcf"
    data_folder.mkdir(parents=True, exist_ok=True)
    for url in URLS:
        download_and_unzip(url, data_folder)
    return 0


if __name__ == "__main__":
    sys.exit(main())

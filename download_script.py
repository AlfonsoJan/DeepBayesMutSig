#!/usr/bin/env python3
import os
import urllib.request
import shutil
from pathlib import Path

# Get the path of the current Python script
script_path = Path(__file__).parent
# Create the data folder if it doesn't exist
data_folder = script_path / "data" / "vcf"
data_folder.mkdir(parents=True, exist_ok=True)

urls = [
    "https://dcc.icgc.org/api/v1/download?fn=/PCAWG/mutational_signatures/Input_Data_PCAWG7_23K_Spectra_DB/vcf_like_simple_files/WES_Other.20180327.simple.gz",
    "https://dcc.icgc.org/api/v1/download?fn=/PCAWG/mutational_signatures/Input_Data_PCAWG7_23K_Spectra_DB/vcf_like_simple_files/WGS_Other.20180413.simple.gz"
]

# Function to download and unzip files
def download_and_unzip(url, output_file, folder: Path = Path(".")):
    download_path = folder / f"{output_file}"
    print(f"Downloading {url} to {download_path}...")
    urllib.request.urlretrieve(url, download_path)
    
    print(f"Unzipping {output_file} to {folder}...")
    shutil.unpack_archive(download_path, folder)

    print(f"Unzipping {output_file}...")
    shutil.unpack_archive(f"{output_file}.gz", ".")

    print(f"Cleaning up...")
    os.remove(f"{output_file}.gz")

    print(f"File {output_file} is done!")



for url in urls:
    output_file = os.path.basename(url.split("?")[1].split("=")[1])
    download_and_unzip(url, output_file, data_folder)
    break

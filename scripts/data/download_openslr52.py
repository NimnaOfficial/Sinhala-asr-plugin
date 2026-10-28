#!/usr/bin/env python3
"""
OpenSLR-52 Dataset Downloader & Ingestion Script
================================================
Downloads and extracts the OpenSLR-52 (Large Sinhala ASR Corpus).
Source: https://www.openslr.org/52/
"""

import os
import sys
import urllib.request
import tarfile
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_openslr52(output_dir: str = "data/raw/openslr52"):
    """Downloads OpenSLR-52 audio archive and transcripts."""
    url = "https://www.openslr.org/resources/52/asr_sinhala.tar.gz"
    dest_path = Path(output_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    tar_path = dest_path / "asr_sinhala.tar.gz"

    if not tar_path.exists():
        print(f"Downloading OpenSLR-52 from {url} to {tar_path}...")
        try:
            with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc="OpenSLR-52") as t:
                urllib.request.urlretrieve(url, filename=tar_path, reporthook=t.update_to)
            print("Download completed successfully.")
        except Exception as e:
            print(f"Error downloading OpenSLR-52: {e}")
            print("You can manually place asr_sinhala.tar.gz into:", output_dir)
            return
    else:
        print(f"Archive already exists at: {tar_path}")

    print("Extracting OpenSLR-52 archive...")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=dest_path)
        print(f"Extraction complete. Data stored in: {dest_path}")
    except Exception as e:
        print(f"Error extracting archive: {e}")


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw/openslr52"
    print("=== OpenSLR-52 Ingestion Utility ===")
    print(f"Target Directory: {out_dir}")
    print("To trigger download, run: python scripts/data/download_openslr52.py [output_path]")

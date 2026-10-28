#!/usr/bin/env python3
"""
OpenSLR-52 Dataset Downloader & Ingestion Script
================================================
Downloads OpenSLR-52 Sinhala ASR Corpus (Transcripts and Audio).
Mirrors: trmal.net, elda.org, magicdatatech.com
"""

import os
import sys
import argparse
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm


MIRRORS = [
    "https://openslr.trmal.net/resources/52/",
    "https://openslr.elda.org/resources/52/",
    "https://openslr.magicdatatech.com/resources/52/",
]


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_with_mirrors(filename: str, output_path: Path) -> bool:
    """Tries downloading a file from multiple mirrors."""
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"File already exists: {output_path}")
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".part")

    for base_url in MIRRORS:
        url = base_url + filename
        print(f"Connecting to mirror: {url}...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as response:
                total_size = int(response.headers.get("Content-Length", 0))
                with open(temp_path, "wb") as out_file, DownloadProgressBar(
                    unit="B", unit_scale=True, miniters=1, desc=filename, total=total_size
                ) as pbar:
                    while True:
                        chunk = response.read(64 * 1024)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        pbar.update(len(chunk))

            # Move temp file to final location
            temp_path.rename(output_path)
            print(f"Downloaded successfully: {output_path}")
            return True
        except Exception as e:
            print(f"Mirror failed ({url}): {e}")
            if temp_path.exists():
                temp_path.unlink()

    print(f"Error: Failed to download {filename} from all mirrors.")
    return False


def ingest_openslr52(output_dir: str = "data/raw/openslr52", download_audio_parts: int = 1):
    """
    Downloads transcripts (15.4 MB) and optionally the first N audio parts (870 MB each).
    """
    dest_dir = Path(output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download utt_spk_text.tsv (Full Transcripts)
    tsv_filename = "utt_spk_text.tsv"
    tsv_path = dest_dir / tsv_filename
    print("\n--- Phase 1: Downloading OpenSLR-52 Transcripts (15.4 MB) ---")
    success = download_with_mirrors(tsv_filename, tsv_path)
    if not success:
        sys.exit(1)

    # 2. Download requested audio archives (0 to f)
    if download_audio_parts > 0:
        print(f"\n--- Phase 2: Downloading Audio Archives ({download_audio_parts} part(s)) ---")
        parts = [f"asr_sinhala_{hex(i)[2:]}.zip" for i in range(min(download_audio_parts, 16))]
        for zip_file in parts:
            zip_path = dest_dir / zip_file
            if download_with_mirrors(zip_file, zip_path):
                extract_target = dest_dir / "audio"
                extract_target.mkdir(parents=True, exist_ok=True)
                print(f"Extracting {zip_file} to {extract_target}...")
                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall(extract_target)
                print(f"Extracted {zip_file} successfully.")

    print("\nOpenSLR-52 Ingestion Completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and ingest OpenSLR-52 Sinhala Dataset")
    parser.add_argument("--output-dir", type=str, default="data/raw/openslr52", help="Target directory")
    parser.add_argument("--audio-parts", type=int, default=1, help="Number of audio zip parts to download (0 to 16, default: 1)")
    parser.add_argument("--transcripts-only", action="store_true", help="Download only the transcripts (utt_spk_text.tsv)")
    args = parser.parse_args()

    parts = 0 if args.transcripts_only else args.audio_parts
    ingest_openslr52(args.output_dir, download_audio_parts=parts)

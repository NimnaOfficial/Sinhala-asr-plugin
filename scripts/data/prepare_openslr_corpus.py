#!/usr/bin/env python3
"""
OpenSLR-52 Corpus Normalization & Preparation
=============================================
Processes all 185,293 utterances from OpenSLR-52:
1. Parses TSV (utterance_id, speaker_id, text).
2. Applies SinhalaNormalizer (NFC, ZWJ ligatures, pili, virama).
3. Filters empty lines or corrupted byte sequences.
4. Outputs clean, normalized corpus ready for 4,096-token Unigram training.
"""

import sys
import unicodedata
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from data.sinhala_normalizer import SinhalaNormalizer


def is_sinhala_or_code_switch(text: str) -> bool:
    """Checks if string contains Sinhala Unicode characters (U+0D80 to U+0DFF)."""
    return any(0x0D80 <= ord(c) <= 0x0DFF for c in text)


def prepare_corpus(
    tsv_path: str = "data/raw/openslr52/utt_spk_text.tsv",
    output_path: str = "data/processed/openslr52_normalized.txt",
):
    tsv_file = Path(tsv_path)
    if not tsv_file.exists():
        print(f"Error: {tsv_file} does not exist. Run download_openslr52.py first.")
        sys.exit(1)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    normalizer = SinhalaNormalizer()

    total_lines = 0
    sinhala_lines = 0
    empty_or_skipped = 0

    print(f"Reading and normalizing transcripts from: {tsv_file}...")
    with open(tsv_file, "r", encoding="utf-8", errors="ignore") as in_f, open(
        out_file, "w", encoding="utf-8"
    ) as out_f:
        for line in tqdm(in_f, desc="Normalizing Sinhala Corpus", unit="lines"):
            total_lines += 1
            parts = line.strip().split("\t")
            if len(parts) < 3:
                empty_or_skipped += 1
                continue

            raw_text = parts[2].strip()
            if not raw_text:
                empty_or_skipped += 1
                continue

            # Check if line contains Sinhala
            if not is_sinhala_or_code_switch(raw_text):
                empty_or_skipped += 1
                continue

            # Apply full normalization pipeline
            normalized = normalizer.normalize(raw_text)
            if normalized:
                out_f.write(normalized + "\n")
                sinhala_lines += 1

    print("\n--- Corpus Normalization Summary ---")
    print(f"Total TSV Lines:         {total_lines}")
    print(f"Clean Sinhala Sentences: {sinhala_lines}")
    print(f"Skipped / Non-Sinhala:   {empty_or_skipped}")
    print(f"Normalized Corpus File:  {out_file} ({out_file.stat().st_size / (1024*1024):.2f} MB)")
    return out_file


if __name__ == "__main__":
    tsv = sys.argv[1] if len(sys.argv) > 1 else "data/raw/openslr52/utt_spk_text.tsv"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/processed/openslr52_normalized.txt"
    prepare_corpus(tsv, out)

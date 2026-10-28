#!/usr/bin/env python3
"""
Sinhala Text Normalization Engine
=================================
Pillar 1 component for the Sinhala ASR & Translation Pipeline.

Handles:
1. Canonical Unicode Normalization (NFC via unicodedata).
2. Valid ZWJ (Zero-Width Joiner) handling for:
   - Rakaransaya (Consonant + Halanta + ZWJ + Ra) -> e.g., ක්‍ර
   - Yansaya (Consonant + Halanta + ZWJ + Ya) -> e.g., ක්‍ය
   - Repaya (Ra + Halanta + ZWJ + Consonant) -> e.g., ර්‍ක
3. Stripping erroneous / dangling ZWJ and ZWNJ characters that corrupt tokenization.
4. Normalizing consecutive / duplicate vowel diacritics (Pili) and Virama (Al-Lakuna).
5. Preserving prenasalized voiced stops (ඟ, ඬ, ඳ, ඹ) as unitary phonemes.
"""

import re
import unicodedata
from typing import Optional


class SinhalaNormalizer:
    """Production-grade text normalizer for Sinhala ASR & NLP pipelines."""

    # Sinhala Unicode Block: U+0D80 to U+0DFF
    SINHALA_RANGE = (0x0D80, 0x0DFF)
    
    # Specific Unicode Codepoints
    HALANTA = "\u0DCA"       # Virama (Al-Lakuna) ්
    ZWJ = "\u200D"           # Zero-Width Joiner
    ZWNJ = "\u200C"          # Zero-Width Non-Joiner
    RA = "\u0DBB"            # Consonant Ra ර
    YA = "\u0DBA"            # Consonant Ya ය

    # Prenasalized consonants (should remain intact)
    PRENASALIZED = {
        "\u0D9F": "ඟ",  # ngga
        "\u0DA6": "ඬ",  # nnda
        "\u0DB3": "ඳ",  # nnda (dental)
        "\u0DB9": "ඹ",  # mmba
    }

    # All Sinhala consonants (U+0D9A to U+0DC6)
    CONSONANTS = r"[\u0D9A-\u0DC6]"
    
    # All Sinhala vowel diacritics (Pili: U+0DCF to U+0DF3)
    PILI = r"[\u0DCF-\u0DF3]"

    def __init__(self, preserve_punctuation: bool = True):
        self.preserve_punctuation = preserve_punctuation
        self._compile_regexes()

    def _compile_regexes(self):
        # 1. Valid Rakaransaya: Consonant + Halanta + ZWJ + Ra
        self.re_valid_rakaransaya = re.compile(
            f"({self.CONSONANTS}){self.HALANTA}{self.ZWJ}{self.RA}"
        )

        # 2. Valid Yansaya: Consonant + Halanta + ZWJ + Ya
        self.re_valid_yansaya = re.compile(
            f"({self.CONSONANTS}){self.HALANTA}{self.ZWJ}{self.YA}"
        )

        # 3. Valid Repaya: Ra + Halanta + ZWJ + Consonant
        self.re_valid_repaya = re.compile(
            f"{self.RA}{self.HALANTA}{self.ZWJ}({self.CONSONANTS})"
        )

        # 4. Duplicate diacritics (e.g. double pili like ීී or ්්)
        self.re_duplicate_diacritics = re.compile(
            r"([\u0DCA-\u0DF3])\1+"
        )

        # 5. Consecutive whitespace
        self.re_whitespace = re.compile(r"[ \t\u00A0]+")

    def normalize(self, text: str) -> str:
        """Applies full normalization pipeline to input string."""
        if not text:
            return ""

        # Step 1: Canonical Decomposition followed by Canonical Composition (NFC)
        text = unicodedata.normalize("NFC", text)

        # Step 2: Remove duplicate diacritics
        text = self.re_duplicate_diacritics.sub(r"\1", text)

        # Step 3: Protect valid ZWJ constructs by temporarily masking them
        # Placeholders for valid ligatures:
        rakaransaya_matches = []
        def _mask_rakaransaya(m):
            rakaransaya_matches.append(m.group(0))
            return f"__RAKARANSAYA_{len(rakaransaya_matches)-1}__"

        yansaya_matches = []
        def _mask_yansaya(m):
            yansaya_matches.append(m.group(0))
            return f"__YANSAYA_{len(yansaya_matches)-1}__"

        repaya_matches = []
        def _mask_repaya(m):
            repaya_matches.append(m.group(0))
            return f"__REPAYA_{len(repaya_matches)-1}__"

        # Protect
        text = self.re_valid_rakaransaya.sub(_mask_rakaransaya, text)
        text = self.re_valid_yansaya.sub(_mask_yansaya, text)
        text = self.re_valid_repaya.sub(_mask_repaya, text)

        # Step 4: Strip all remaining (orphan/dangling) ZWJ and ZWNJ
        text = text.replace(self.ZWJ, "").replace(self.ZWNJ, "")

        # Step 5: Restore protected valid ligatures
        for i, val in enumerate(rakaransaya_matches):
            text = text.replace(f"__RAKARANSAYA_{i}__", val)
        for i, val in enumerate(yansaya_matches):
            text = text.replace(f"__YANSAYA_{i}__", val)
        for i, val in enumerate(repaya_matches):
            text = text.replace(f"__REPAYA_{i}__", val)

        # Step 6: Normalize whitespace
        text = self.re_whitespace.sub(" ", text).strip()

        return text


# Quick CLI test
if __name__ == "__main__":
    normalizer = SinhalaNormalizer()
    samples = [
        "ක්‍රියාකාරකම්",          # Rakaransaya (valid)
        "වාක්‍ය",                 # Yansaya (valid)
        "ධර්‍මය",                 # Repaya (valid)
        "ශ්‍රී\u200D\u200D ලංකා", # Duplicate ZWJs
        "පොතක්   කියවන්න",       # Spacing
    ]

    print("=== Sinhala Text Normalizer Verification ===")
    for s in samples:
        norm = normalizer.normalize(s)
        print(f"Original: {s} -> Normalized: {norm}")

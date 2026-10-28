#!/usr/bin/env python3
"""
Unit Tests for Sinhala Text Normalizer
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.data.sinhala_normalizer import SinhalaNormalizer


def test_rakaransaya_preservation():
    norm = SinhalaNormalizer()
    word = "ක්‍රියාකාරකම්"
    result = norm.normalize(word)
    assert result == word, f"Expected {word}, got {result}"
    print("PASS: Rakaransaya preservation")


def test_yansaya_preservation():
    norm = SinhalaNormalizer()
    word = "සංඛ්‍යා"
    result = norm.normalize(word)
    assert result == word, f"Expected {word}, got {result}"
    print("PASS: Yansaya preservation")


def test_repaya_preservation():
    norm = SinhalaNormalizer()
    word = "ධර්‍මය"
    result = norm.normalize(word)
    assert result == word, f"Expected {word}, got {result}"
    print("PASS: Repaya preservation")


def test_prenasalized_preservation():
    norm = SinhalaNormalizer()
    words = ["ඟඟ", "කැඳ", "බිම්බ", "හඬ"]
    for w in words:
        assert norm.normalize(w) == w
    print("PASS: Prenasalized consonants preservation")


def test_dangling_zwj_removal():
    norm = SinhalaNormalizer()
    dirty = "ශ්‍රී\u200D ලංකා\u200C"
    expected = "ශ්‍රී ලංකා"
    result = norm.normalize(dirty)
    assert result == expected, f"Expected '{expected}', got '{result}'"
    print("PASS: Dangling ZWJ/ZWNJ removal")


def test_duplicate_diacritics_removal():
    norm = SinhalaNormalizer()
    dirty = "පොතක\u0DCA\u0DCA"  # duplicate halanta virama: ka + virama + virama
    expected = "පොතක්"
    result = norm.normalize(dirty)
    assert result == expected, f"Expected '{expected}', got '{result}'"
    
    # Duplicate pili: ka + ispilla + ispilla
    dirty_pili = "ක\u0DD2\u0DD2"
    expected_pili = "කි"
    assert norm.normalize(dirty_pili) == expected_pili
    print("PASS: Duplicate diacritics removal")


if __name__ == "__main__":
    test_rakaransaya_preservation()
    test_yansaya_preservation()
    test_repaya_preservation()
    test_prenasalized_preservation()
    test_dangling_zwj_removal()
    test_duplicate_diacritics_removal()
    print("All Sinhala normalizer tests passed successfully!")

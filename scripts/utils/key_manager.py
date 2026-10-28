#!/usr/bin/env python3
"""
Developer Key Manager (Offline Dev-Only Utility)
================================================
Manages automatic rotation across the 20 free-tier API keys in apis.txt.
Handles HTTP 429 (rate limits) and quota exhaustion by failing over to the next key.

NOTE: This is strictly an offline developer utility for assistance and research.
It is NEVER imported or used by the runtime plugin or model inference pipeline.
"""

import os
from pathlib import Path
from typing import List, Optional


class DevKeyManager:
    """Rotates free-tier API keys for lightweight developer queries."""

    def __init__(self, key_file_path: Optional[str] = None):
        if key_file_path is None:
            # Look in project root
            root_dir = Path(__file__).resolve().parent.parent.parent
            key_file_path = root_dir / "apis.txt"
        
        self.key_file_path = Path(key_file_path)
        self.keys: List[str] = []
        self.current_index: int = 0
        self._load_keys()

    def _load_keys(self):
        """Loads API keys from apis.txt, skipping comments and empty lines."""
        if not self.key_file_path.exists():
            return

        with open(self.key_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and not line.lower().startswith("google"):
                    self.keys.append(line)

    @property
    def total_keys(self) -> int:
        return len(self.keys)

    def get_current_key(self) -> Optional[str]:
        """Returns the current active key."""
        if not self.keys:
            return None
        return self.keys[self.current_index]

    def rotate_to_next_key(self) -> Optional[str]:
        """Rotates to the next available key when a rate limit or 429 error occurs."""
        if not self.keys:
            return None
        self.current_index = (self.current_index + 1) % len(self.keys)
        return self.keys[self.current_index]


if __name__ == "__main__":
    km = DevKeyManager()
    print(f"Loaded {km.total_keys} developer API keys for round-robin rotation.")

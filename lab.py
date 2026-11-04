#!/usr/bin/env python3
"""Source-checkout entry point: python lab.py doctor / train / serve."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from training_lab.cli import main

if __name__ == "__main__":
    main()

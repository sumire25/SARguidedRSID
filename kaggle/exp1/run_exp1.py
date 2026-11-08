#!/usr/bin/env python3
"""Kaggle runner entrypoint for Experiment 1: Frequency-Decoupled SAR Guidance."""
import sys
from pathlib import Path

# Add directory containing worker.py to python path
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

from worker import main

if __name__ == "__main__":
    sys.argv = [
        "worker.py",
        "--branch", "exp1-freq-decoupled",
        "--config", "configs/exp1_freq_decoupled.yaml",
    ]
    main()

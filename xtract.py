#!/usr/bin/env python3
"""
PDXTRACT launcher wrapper.

This file exists for backward compatibility with the original project layout.
It simply delegates to the modular `pdxtract` package.

Usage:
    python xtract.py
    python xtract.py process --help
    python xtract.py list-models
"""

import sys

from pdxtract.cli import main

if __name__ == "__main__":
    sys.exit(main())

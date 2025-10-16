#!/usr/bin/env python3
"""
Test Script - Quick non-interactive test of the system
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import src.test_query as test_module


if __name__ == "__main__":
    test_module.main()


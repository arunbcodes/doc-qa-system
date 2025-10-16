#!/usr/bin/env python3
"""
PDF Q&A System - RAG Entry Point (Phase 2: Retrieval + Generation)
Processes PDF and provides LLM-powered natural language answers.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import from src using module syntax to avoid issues
import src.main_rag as rag_module


if __name__ == "__main__":
    # Just delegate to the module's main function
    rag_module.main()

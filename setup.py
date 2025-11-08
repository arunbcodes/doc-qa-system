#!/usr/bin/env python
"""
Setup script for backward compatibility.
Modern installations should use pyproject.toml with pip >= 21.0
"""

from setuptools import setup

# All configuration is in pyproject.toml
# This file exists for backward compatibility with older pip versions
setup()

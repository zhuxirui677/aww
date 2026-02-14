#!/usr/bin/env python3
"""
Compatibility entrypoint for MPS test script.
"""

import os
import runpy


def main():
    here = os.path.dirname(__file__)
    target = os.path.join(here, "scripts", "test_medgamma_clean_mps.py")
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()

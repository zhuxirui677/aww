#!/usr/bin/env python3
"""
兼容入口：实际脚本在 scripts/quantize_medgamma_awq_w4a4_w4a8.py
"""

import os
import runpy


def main():
    here = os.path.dirname(__file__)
    target = os.path.join(here, "scripts", "quantize_medgamma_awq_w4a4_w4a8.py")
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()

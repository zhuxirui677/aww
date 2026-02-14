#!/usr/bin/env python3
"""
兼容入口：实际脚本已迁移到 scripts/quantize_medgamma_awq.py
"""

import os
import runpy


def main():
    here = os.path.dirname(__file__)
    target = os.path.join(here, "scripts", "quantize_medgamma_awq.py")
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()

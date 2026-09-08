#!/usr/bin/env python
"""Development scaling entry point; held-out uses the unified all-suite run."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]/'src'))
from ccl.stage3_eval import main
if __name__=='__main__':
    raise SystemExit(main(default_suite='scaling'))

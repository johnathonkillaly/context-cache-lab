#!/usr/bin/env python
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]/'src'))
from ccl.stage3_eval import main
if __name__=='__main__':
    raise SystemExit(main())

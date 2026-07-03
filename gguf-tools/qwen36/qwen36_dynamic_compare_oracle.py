#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path


def main() -> int:
    target = Path(__file__).with_name("qwen36_hybrid_prefix_tail_greedy.py")
    cmd = [sys.executable, str(target), *sys.argv[1:]]
    print(f"[dynamic-oracle] delegating to {target}")
    print(f"[dynamic-oracle] cmd={' '.join(cmd)}")
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

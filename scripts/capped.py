"""Run ``python -m <module> args...`` with guard rails for a shared 8 GB laptop GPU.

PyTorch is capped at 85% of VRAM so an oversized job raises OOM instead of spilling into
system RAM, and the process (and its dataloader workers) runs at below-normal CPU priority.
Usage: ``python scripts/capped.py evaluation.detection --models yolo26s``.
"""

import os
import runpy
import sys
from pathlib import Path

import psutil
import torch

if torch.cuda.is_available():  # VRAM_FRACTION lets several light jobs share the GPU safely
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("VRAM_FRACTION", "0.85")))
if sys.platform == "win32":
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
sys.path[0] = str(Path(__file__).resolve().parent.parent)  # like `python -m` from the root
sys.argv = sys.argv[1:]
runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)

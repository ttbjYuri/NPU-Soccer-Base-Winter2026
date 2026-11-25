"""pytest 入口占位：确保 ``import roadseg`` 能在仓库根目录直接跑。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

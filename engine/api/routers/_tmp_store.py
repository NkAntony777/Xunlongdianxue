"""临时上传目录共享工具。"""
from __future__ import annotations
import time
from pathlib import Path

TMP_DIR = Path("data/_tmp_uploads")
TMP_DIR.mkdir(parents=True, exist_ok=True)

def new_tmp_name(prefix: str, suffix: str) -> Path:
    return TMP_DIR / f"{prefix}_{int(time.time()*1000)}{suffix}"

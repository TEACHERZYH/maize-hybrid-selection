"""稳定的任务子流，不依赖Python进程hash。"""
import hashlib
import json
import numpy as np


def stream(protocol, year, seed, purpose, object_id, *extra):
    raw = json.dumps([protocol, year, seed, purpose, object_id, *extra], ensure_ascii=False,
                     separators=(',', ':'), allow_nan=False).encode('utf-8')
    value = int.from_bytes(hashlib.sha256(raw).digest()[:8], 'little')
    return np.random.Generator(np.random.PCG64(value))

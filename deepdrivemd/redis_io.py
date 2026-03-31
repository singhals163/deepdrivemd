"""Redis I/O utilities for DeepDriveMD data passing experiment.

Provides instrumented store/load functions for numpy arrays and torch
checkpoints via Redis, with per-operation timing and size statistics.
"""
import io
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Module-level singleton connection, lazy-initialized per worker process
_redis_client = None
_redis_config = {"host": None, "port": None}

# Accumulated per-operation timing and size statistics
_io_stats: Dict[str, list] = defaultdict(list)


def init_redis(host: str = "127.0.0.1", port: int = 6379) -> None:
    """Establish Redis connection, cached per process."""
    global _redis_client, _redis_config
    if _redis_client is not None and _redis_config == {"host": host, "port": port}:
        return
    import redis

    _redis_client = redis.Redis(host=host, port=port, decode_responses=False)
    _redis_config["host"] = host
    _redis_config["port"] = port
    _redis_client.ping()
    logger.info(f"Redis connection established: {host}:{port}")


def _get_client():
    if _redis_client is None:
        raise RuntimeError("Redis not initialized. Call init_redis() first.")
    return _redis_client


def store_numpy(path, array) -> Dict[str, float]:
    """Serialize numpy array to bytes and store in Redis.

    Uses the string representation of `path` as the Redis key.
    Returns timing dict with serialize_s, transfer_s, data_bytes.
    """
    client = _get_client()
    key = str(path)

    t0 = time.perf_counter()
    buf = io.BytesIO()
    np.save(buf, array, allow_pickle=True)
    data = buf.getvalue()
    t1 = time.perf_counter()

    client.set(key, data)
    t2 = time.perf_counter()

    stats = {
        "op": "store_numpy",
        "key": key,
        "serialize_s": t1 - t0,
        "transfer_s": t2 - t1,
        "data_bytes": len(data),
    }
    _io_stats["store_numpy"].append(stats)
    return stats


def load_numpy(path, allow_pickle: bool = False) -> Optional[np.ndarray]:
    """Load numpy array from Redis.

    Returns None if key is not found in Redis.
    Returns timing dict info via side-effect in _io_stats.
    """
    client = _get_client()
    key = str(path)

    t0 = time.perf_counter()
    data = client.get(key)
    t1 = time.perf_counter()

    if data is None:
        return None

    buf = io.BytesIO(data)
    result = np.load(buf, allow_pickle=allow_pickle)
    t2 = time.perf_counter()

    stats = {
        "op": "load_numpy",
        "key": key,
        "transfer_s": t1 - t0,
        "deserialize_s": t2 - t1,
        "data_bytes": len(data),
    }
    _io_stats["load_numpy"].append(stats)
    return result


def store_torch(path, obj) -> Dict[str, float]:
    """Serialize torch object to bytes and store in Redis.

    Uses the string representation of `path` as the Redis key.
    Returns timing dict with serialize_s, transfer_s, data_bytes.
    """
    import torch

    client = _get_client()
    key = str(path)

    t0 = time.perf_counter()
    buf = io.BytesIO()
    torch.save(obj, buf)
    data = buf.getvalue()
    t1 = time.perf_counter()

    client.set(key, data)
    t2 = time.perf_counter()

    stats = {
        "op": "store_torch",
        "key": key,
        "serialize_s": t1 - t0,
        "transfer_s": t2 - t1,
        "data_bytes": len(data),
    }
    _io_stats["store_torch"].append(stats)
    return stats


def load_torch(path, map_location=None) -> Optional[Any]:
    """Load torch object from Redis.

    Returns None if key is not found in Redis.
    """
    import torch

    client = _get_client()
    key = str(path)

    t0 = time.perf_counter()
    data = client.get(key)
    t1 = time.perf_counter()

    if data is None:
        return None

    buf = io.BytesIO(data)
    result = torch.load(buf, map_location=map_location)
    t2 = time.perf_counter()

    stats = {
        "op": "load_torch",
        "key": key,
        "transfer_s": t1 - t0,
        "deserialize_s": t2 - t1,
        "data_bytes": len(data),
    }
    _io_stats["load_torch"].append(stats)
    return result


def get_io_stats() -> Dict[str, list]:
    """Return accumulated per-operation timing and size statistics."""
    return dict(_io_stats)


def reset_io_stats() -> None:
    """Clear accumulated statistics (call at start of each task)."""
    _io_stats.clear()


def write_io_stats(workdir: Path) -> None:
    """Write accumulated stats to JSON file and log summary."""
    stats = get_io_stats()
    if not stats:
        return

    output_path = workdir / "redis_io_stats.json"
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Log summary
    for op_name, entries in stats.items():
        total_bytes = sum(e.get("data_bytes", 0) for e in entries)
        total_time = sum(
            e.get("serialize_s", 0) + e.get("transfer_s", 0) + e.get("deserialize_s", 0)
            for e in entries
        )
        logger.info(
            f"Redis I/O summary [{op_name}]: {len(entries)} ops, "
            f"{total_bytes / 1024:.1f} KB, {total_time:.4f}s total"
        )

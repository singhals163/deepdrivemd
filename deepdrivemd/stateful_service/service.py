"""StatefulService — model lifecycle management via Redis backing store.

Wraps redis_io.store_torch/load_torch with lifecycle semantics:
- serialize(): save current model checkpoint to Redis (on DORMANT transition)
- deserialize(): restore model checkpoint from Redis (on RESUME transition)
- get_stats(): return timing/size statistics for all operations

Workers are still stateless — they create a fresh model each call and
load weights from Redis or disk. The StatefulService manages the backing
store so that warm-start loads from Redis (~10ms) instead of filesystem.
"""
import io
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class StatefulService:
    """Model state lifecycle manager backed by Redis.

    Parameters
    ----------
    redis_host : str
        Redis server hostname.
    redis_port : int
        Redis server port.
    key_prefix : str
        Prefix for Redis keys to namespace model state.
    """

    def __init__(
        self,
        redis_host: str = "127.0.0.1",
        redis_port: int = 6379,
        key_prefix: str = "deepdrivemd:model",
    ):
        self._redis_host = redis_host
        self._redis_port = redis_port
        self._key_prefix = key_prefix
        self._initialized = False
        self._operation_log: List[Dict[str, Any]] = []

    def _ensure_redis(self) -> None:
        """Lazy-initialize Redis connection."""
        if not self._initialized:
            from deepdrivemd.redis_io import init_redis
            init_redis(self._redis_host, self._redis_port)
            self._initialized = True

    def _make_key(self, name: str) -> str:
        return f"{self._key_prefix}:{name}"

    def serialize(self, checkpoint_path: Path, key_name: str = "weights") -> Dict[str, float]:
        """Serialize a model checkpoint file to Redis.

        Reads the checkpoint from disk and stores it in Redis for fast
        retrieval by future workers.

        Parameters
        ----------
        checkpoint_path : Path
            Path to the torch checkpoint file on disk.
        key_name : str
            Logical name for this checkpoint in Redis.

        Returns
        -------
        dict
            Timing statistics: serialize_s, transfer_s, data_bytes.
        """
        import torch
        from deepdrivemd.redis_io import store_torch

        self._ensure_redis()
        key = self._make_key(key_name)

        t0 = time.perf_counter()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        t_load = time.perf_counter()

        stats = store_torch(key, checkpoint)
        t_done = time.perf_counter()

        record = {
            "op": "serialize",
            "key": key,
            "source": str(checkpoint_path),
            "disk_load_s": t_load - t0,
            "total_s": t_done - t0,
            **stats,
        }
        self._operation_log.append(record)
        logger.info(
            f"Serialized {checkpoint_path} -> Redis[{key}] "
            f"({stats['data_bytes'] / 1024:.1f} KB, {t_done - t0:.4f}s)"
        )
        return record

    def serialize_object(self, obj: Any, key_name: str = "weights") -> Dict[str, float]:
        """Serialize a torch object directly to Redis (no disk read).

        Parameters
        ----------
        obj : Any
            Torch-serializable object (state_dict, model, etc.).
        key_name : str
            Logical name for this checkpoint in Redis.

        Returns
        -------
        dict
            Timing statistics.
        """
        from deepdrivemd.redis_io import store_torch

        self._ensure_redis()
        key = self._make_key(key_name)

        t0 = time.perf_counter()
        stats = store_torch(key, obj)
        t_done = time.perf_counter()

        record = {
            "op": "serialize_object",
            "key": key,
            "total_s": t_done - t0,
            **stats,
        }
        self._operation_log.append(record)
        logger.info(
            f"Serialized object -> Redis[{key}] "
            f"({stats['data_bytes'] / 1024:.1f} KB, {t_done - t0:.4f}s)"
        )
        return record

    def deserialize(self, key_name: str = "weights", map_location=None) -> Optional[Any]:
        """Deserialize a model checkpoint from Redis.

        Parameters
        ----------
        key_name : str
            Logical name of the checkpoint in Redis.
        map_location : optional
            torch.load map_location argument (e.g. "cpu", "cuda:0").

        Returns
        -------
        Any or None
            The deserialized torch object, or None if not found.
        """
        from deepdrivemd.redis_io import load_torch

        self._ensure_redis()
        key = self._make_key(key_name)

        t0 = time.perf_counter()
        result = load_torch(key, map_location=map_location)
        t_done = time.perf_counter()

        record = {
            "op": "deserialize",
            "key": key,
            "total_s": t_done - t0,
            "found": result is not None,
        }
        self._operation_log.append(record)

        if result is not None:
            logger.info(f"Deserialized Redis[{key}] ({t_done - t0:.4f}s)")
        else:
            logger.warning(f"Key {key} not found in Redis")

        return result

    def has_checkpoint(self, key_name: str = "weights") -> bool:
        """Check whether a checkpoint exists in Redis."""
        self._ensure_redis()
        from deepdrivemd.redis_io import _get_client
        key = self._make_key(key_name)
        return _get_client().exists(key) > 0

    def delete(self, key_name: str = "weights") -> bool:
        """Remove a checkpoint from Redis."""
        self._ensure_redis()
        from deepdrivemd.redis_io import _get_client
        key = self._make_key(key_name)
        deleted = _get_client().delete(key)
        logger.info(f"Deleted Redis[{key}]: {'found' if deleted else 'not found'}")
        return deleted > 0

    def get_stats(self) -> Dict[str, Any]:
        """Return operation log and summary statistics."""
        if not self._operation_log:
            return {"operations": [], "summary": {}}

        serialize_ops = [r for r in self._operation_log if r["op"].startswith("serialize")]
        deserialize_ops = [r for r in self._operation_log if r["op"] == "deserialize"]

        summary = {
            "total_operations": len(self._operation_log),
            "serialize_count": len(serialize_ops),
            "deserialize_count": len(deserialize_ops),
        }

        if serialize_ops:
            summary["serialize_mean_s"] = sum(r["total_s"] for r in serialize_ops) / len(serialize_ops)
            summary["serialize_total_bytes"] = sum(r.get("data_bytes", 0) for r in serialize_ops)

        if deserialize_ops:
            summary["deserialize_mean_s"] = sum(r["total_s"] for r in deserialize_ops) / len(deserialize_ops)

        return {"operations": self._operation_log, "summary": summary}

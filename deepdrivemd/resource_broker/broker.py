"""ResourceBroker — dynamic GPU reallocation with cooldown and draining.

Coordinates freeze/resume transitions for the ML executor:
- freeze(): reclaim ML GPU for simulations (route sims to htex_ml)
- resume(): restore ML GPU for training/inference
- Cooldown prevents rapid oscillation between states
- Graceful draining waits for in-flight ML tasks before reclaim
- All transitions are timed and logged for analysis
"""
import logging
import threading
import time
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class GPUState(IntEnum):
    """State of the ML GPU allocation."""
    ML_ACTIVE = 0      # GPU running training/inference
    DRAINING = 1       # Waiting for in-flight ML tasks to complete
    SIM_RECLAIMED = 2  # GPU reclaimed for simulation work


class ResourceBroker:
    """Dynamic GPU resource broker with cooldown and graceful draining.

    Parameters
    ----------
    cooldown_sec : float
        Minimum seconds between state transitions to prevent thrashing.
    on_freeze : callable, optional
        Callback invoked when GPU transitions to SIM_RECLAIMED.
        Signature: on_freeze(transition_record: dict) -> None
    on_resume : callable, optional
        Callback invoked when GPU transitions back to ML_ACTIVE.
        Signature: on_resume(transition_record: dict) -> None
    """

    def __init__(
        self,
        cooldown_sec: float = 30.0,
        on_freeze: Optional[Callable[[Dict], None]] = None,
        on_resume: Optional[Callable[[Dict], None]] = None,
    ):
        self._cooldown_sec = cooldown_sec
        self._on_freeze = on_freeze
        self._on_resume = on_resume
        self._lock = threading.Lock()
        self._state = GPUState.ML_ACTIVE
        self._last_transition_time = 0.0
        self._transition_log: List[Dict[str, Any]] = []
        self._in_flight_ml_tasks = 0
        self._freeze_requested_at: Optional[float] = None

    @property
    def state(self) -> GPUState:
        return self._state

    @property
    def in_flight_ml_tasks(self) -> int:
        return self._in_flight_ml_tasks

    def register_ml_task_start(self) -> None:
        """Track an ML task starting on the GPU."""
        with self._lock:
            self._in_flight_ml_tasks += 1

    def register_ml_task_complete(self) -> None:
        """Track an ML task completing. If draining and no tasks remain, finalize freeze."""
        with self._lock:
            self._in_flight_ml_tasks = max(0, self._in_flight_ml_tasks - 1)
            if self._state == GPUState.DRAINING and self._in_flight_ml_tasks == 0:
                self._finalize_freeze()

    def freeze(self, workflow: Any = None) -> bool:
        """Request GPU reclaim for simulation work.

        If ML tasks are in-flight, enters DRAINING state and waits.
        If no tasks are in-flight, transitions directly to SIM_RECLAIMED.

        Parameters
        ----------
        workflow : optional
            The workflow instance, used to call simulate_on_ml() on reclaim.

        Returns
        -------
        bool
            True if freeze was initiated, False if blocked by cooldown.
        """
        with self._lock:
            if self._state != GPUState.ML_ACTIVE:
                logger.debug(f"Freeze ignored: already in {self._state.name}")
                return False

            now = time.perf_counter()
            if now - self._last_transition_time < self._cooldown_sec:
                logger.debug(
                    f"Freeze blocked by cooldown: {now - self._last_transition_time:.1f}s "
                    f"< {self._cooldown_sec}s"
                )
                return False

            self._freeze_requested_at = now

            if self._in_flight_ml_tasks > 0:
                self._state = GPUState.DRAINING
                self._workflow_ref = workflow
                logger.info(
                    f"Freeze requested: DRAINING ({self._in_flight_ml_tasks} in-flight ML tasks)"
                )
                record = self._make_transition_record(
                    GPUState.ML_ACTIVE, GPUState.DRAINING
                )
                self._transition_log.append(record)
                return True
            else:
                self._workflow_ref = workflow
                self._finalize_freeze()
                return True

    def _finalize_freeze(self) -> None:
        """Complete the transition to SIM_RECLAIMED (called under lock)."""
        old_state = self._state
        self._state = GPUState.SIM_RECLAIMED
        self._last_transition_time = time.perf_counter()

        drain_time = None
        if self._freeze_requested_at is not None:
            drain_time = self._last_transition_time - self._freeze_requested_at

        record = self._make_transition_record(old_state, GPUState.SIM_RECLAIMED)
        record["drain_time_s"] = drain_time
        self._transition_log.append(record)

        logger.info(
            f"GPU RECLAIMED for simulations "
            f"(drain_time={drain_time:.3f}s)" if drain_time else
            f"GPU RECLAIMED for simulations"
        )

        if self._on_freeze is not None:
            self._on_freeze(record)

        # Trigger simulation on ML GPU if workflow reference available
        if hasattr(self, '_workflow_ref') and self._workflow_ref is not None:
            if hasattr(self._workflow_ref, 'simulate_on_ml'):
                self._workflow_ref.simulate_on_ml()

    def resume(self, workflow: Any = None) -> bool:
        """Restore GPU for ML training/inference work.

        Parameters
        ----------
        workflow : optional
            The workflow instance.

        Returns
        -------
        bool
            True if resume succeeded, False if blocked by cooldown.
        """
        with self._lock:
            if self._state == GPUState.ML_ACTIVE:
                logger.debug("Resume ignored: already ML_ACTIVE")
                return False

            now = time.perf_counter()
            if now - self._last_transition_time < self._cooldown_sec:
                logger.debug(
                    f"Resume blocked by cooldown: {now - self._last_transition_time:.1f}s "
                    f"< {self._cooldown_sec}s"
                )
                return False

            old_state = self._state
            self._state = GPUState.ML_ACTIVE
            self._last_transition_time = now
            self._freeze_requested_at = None

            record = self._make_transition_record(old_state, GPUState.ML_ACTIVE)
            self._transition_log.append(record)

            logger.info(f"GPU RESTORED for ML training/inference")

            if self._on_resume is not None:
                self._on_resume(record)

            return True

    def _make_transition_record(self, from_state: GPUState, to_state: GPUState) -> Dict[str, Any]:
        return {
            "timestamp": time.perf_counter(),
            "from": from_state.name,
            "to": to_state.name,
            "in_flight_ml_tasks": self._in_flight_ml_tasks,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Return transition log and summary statistics."""
        with self._lock:
            freeze_count = sum(
                1 for t in self._transition_log if t["to"] == "SIM_RECLAIMED"
            )
            resume_count = sum(
                1 for t in self._transition_log if t["to"] == "ML_ACTIVE" and t["from"] != "ML_ACTIVE"
            )
            drain_times = [
                t["drain_time_s"] for t in self._transition_log
                if t.get("drain_time_s") is not None
            ]

            return {
                "current_state": self._state.name,
                "total_transitions": len(self._transition_log),
                "freeze_count": freeze_count,
                "resume_count": resume_count,
                "drain_times_s": drain_times,
                "mean_drain_time_s": sum(drain_times) / len(drain_times) if drain_times else None,
                "cooldown_sec": self._cooldown_sec,
                "transition_log": list(self._transition_log),
            }

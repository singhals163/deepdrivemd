"""SignalMonitor — thread-safe telemetry transport and policy dispatch.

Receives telemetry vectors V_t from AI components, maintains an ordered
window, evaluates the configured policy, and emits state transition
commands to the orchestrator.
"""
import logging
import threading
import time
from collections import deque
from typing import Callable, List, Optional

import numpy as np

from deepdrivemd.signal_monitor.policy import Policy, WorkflowState

logger = logging.getLogger(__name__)


class SignalMonitor:
    """Asynchronous signal monitor with pluggable policy.

    Thread-safe: multiple producers can call ``submit_telemetry``
    concurrently.  The monitor maintains ordering via a sequence
    counter and processes signals in FIFO order.

    Parameters
    ----------
    policy : Policy
        The decision policy to evaluate on each telemetry update.
    window_size : int
        Maximum number of telemetry vectors retained in the window.
    on_state_change : callable, optional
        Callback invoked with (old_state, new_state) on transitions.
    cooldown_sec : float
        Minimum seconds between state transitions to prevent thrashing.
    """

    def __init__(
        self,
        policy: Policy,
        window_size: int = 50,
        on_state_change: Optional[Callable[[WorkflowState, WorkflowState], None]] = None,
        cooldown_sec: float = 0.0,
    ):
        self.policy = policy
        self._window: deque = deque(maxlen=window_size)
        self._lock = threading.Lock()
        self._seq_counter = 0
        self._current_state = WorkflowState.ACTIVE
        self._on_state_change = on_state_change
        self._cooldown_sec = cooldown_sec
        self._last_transition_time = 0.0

        # Statistics
        self.signals_received = 0
        self.transitions_issued = 0
        self.duplicate_transitions = 0
        self.dropped_signals = 0

    @property
    def current_state(self) -> WorkflowState:
        return self._current_state

    def submit_telemetry(self, v_t: np.ndarray, seq: Optional[int] = None) -> WorkflowState:
        """Submit a telemetry vector for evaluation.

        Thread-safe.  Returns the resulting workflow state.

        Parameters
        ----------
        v_t : np.ndarray
            Telemetry vector from the AI component.
        seq : int, optional
            Sequence number for ordering.  Auto-assigned if omitted.

        Returns
        -------
        WorkflowState
            The current workflow state after evaluating the new telemetry.
        """
        eval_start = time.perf_counter()

        with self._lock:
            if seq is None:
                seq = self._seq_counter
            self._seq_counter = max(self._seq_counter, seq) + 1

            self._window.append(v_t)
            self.signals_received += 1

            # Evaluate policy
            window_list: List[np.ndarray] = list(self._window)
            new_state = self.policy.evaluate(window_list)

            # Check for state transition
            if new_state != self._current_state:
                now = time.perf_counter()
                if now - self._last_transition_time >= self._cooldown_sec:
                    old_state = self._current_state
                    self._current_state = new_state
                    self._last_transition_time = now
                    self.transitions_issued += 1

                    if self._on_state_change is not None:
                        self._on_state_change(old_state, new_state)

                    logger.info(
                        f"State transition: {old_state.name} -> {new_state.name} "
                        f"(signal #{self.signals_received})"
                    )
                else:
                    self.duplicate_transitions += 1

        eval_end = time.perf_counter()
        latency_ms = (eval_end - eval_start) * 1000
        logger.debug(f"Signal processed in {latency_ms:.3f} ms")

        return self._current_state

    def get_stats(self) -> dict:
        """Return monitoring statistics."""
        with self._lock:
            return {
                "signals_received": self.signals_received,
                "transitions_issued": self.transitions_issued,
                "duplicate_transitions": self.duplicate_transitions,
                "dropped_signals": self.dropped_signals,
                "current_state": self._current_state.name,
                "window_size": len(self._window),
                "policy": self.policy.name,
            }

"""Signal Monitor: lightweight policy-based decision interface.

The Signal Monitor decouples AI telemetry inspection from the simulation
critical path.  A pluggable *Policy* evaluates a telemetry window and
returns the next workflow state, while the *SignalMonitor* manages
transport, ordering, and dispatch.
"""

from deepdrivemd.signal_monitor.policy import (
    Policy,
    ThresholdPolicy,
    SlidingWindowPolicy,
    MannKendallPolicy,
    WorkflowState,
)
from deepdrivemd.signal_monitor.monitor import SignalMonitor

__all__ = [
    "Policy",
    "ThresholdPolicy",
    "SlidingWindowPolicy",
    "MannKendallPolicy",
    "SignalMonitor",
    "WorkflowState",
]

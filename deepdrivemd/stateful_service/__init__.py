"""Stateful Service: model lifecycle management via Redis backing store.

Manages WHERE model bytes live between stateless Parsl worker calls.
Workers still reload models from scratch each call (stateless), but the
Stateful Service controls whether the backing store is Redis (fast) or
filesystem (slow), and handles serialize/deserialize on state transitions.
"""

from deepdrivemd.stateful_service.service import StatefulService

__all__ = ["StatefulService"]

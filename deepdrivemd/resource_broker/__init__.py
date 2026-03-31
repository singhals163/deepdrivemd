"""Resource Broker: dynamic GPU reallocation between AI and simulation.

Wraps the existing freeze mechanism (simulate_on_ml) with timing
instrumentation, cooldown-based thrashing prevention, and graceful
draining of in-flight ML tasks before GPU reclaim.
"""

from deepdrivemd.resource_broker.broker import ResourceBroker

__all__ = ["ResourceBroker"]

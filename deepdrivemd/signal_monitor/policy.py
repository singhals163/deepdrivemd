"""Pluggable policy interface for the Signal Monitor.

Three concrete policies of increasing computational complexity:
  1. ThresholdPolicy   — O(1) scalar comparison
  2. SlidingWindowPolicy — O(Nk) sliding-window mean
  3. MannKendallPolicy  — O(Nk^2) trend test

All policies implement ``evaluate(W_t) -> S_{t+1}`` where W_t is a
window of recent telemetry vectors and S_{t+1} is the next workflow state.
"""
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import List

import numpy as np


class WorkflowState(IntEnum):
    """Workflow states for the AI component.

    0 = ACTIVE   — AI training/inference running normally
    1 = DORMANT  — AI frozen, GPUs reclaimed for simulation
    2 = RESUME   — AI should be restarted (distribution shift detected)
    """
    ACTIVE = 0
    DORMANT = 1
    RESUME = 2


class Policy(ABC):
    """Abstract base class for signal monitor policies.

    Subclasses implement ``evaluate`` which receives a window of
    telemetry vectors and returns the next workflow state.
    """

    @abstractmethod
    def evaluate(self, telemetry_window: List[np.ndarray]) -> WorkflowState:
        """Evaluate telemetry and decide the next workflow state.

        Parameters
        ----------
        telemetry_window : list[np.ndarray]
            Recent telemetry vectors (most recent last).
            Each vector contains metrics like training loss,
            reconstruction error, novelty ratio, etc.

        Returns
        -------
        WorkflowState
            The recommended next state for the AI component.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable policy name."""
        ...


class ThresholdPolicy(Policy):
    """O(1) — freeze when the latest loss drops below a fixed threshold."""

    def __init__(self, loss_threshold: float = 0.05, loss_index: int = 0):
        self.loss_threshold = loss_threshold
        self.loss_index = loss_index

    @property
    def name(self) -> str:
        return "threshold"

    def evaluate(self, telemetry_window: List[np.ndarray]) -> WorkflowState:
        if not telemetry_window:
            return WorkflowState.ACTIVE
        latest = telemetry_window[-1]
        loss = float(latest[self.loss_index])
        if loss < self.loss_threshold:
            return WorkflowState.DORMANT
        return WorkflowState.ACTIVE


class SlidingWindowPolicy(Policy):
    """O(Nk) — freeze when the mean loss over the last k vectors is below threshold,
    and the improvement rate is below a minimum delta."""

    def __init__(
        self,
        window_size: int = 10,
        loss_threshold: float = 0.05,
        min_improvement: float = 0.001,
        loss_index: int = 0,
    ):
        self.window_size = window_size
        self.loss_threshold = loss_threshold
        self.min_improvement = min_improvement
        self.loss_index = loss_index

    @property
    def name(self) -> str:
        return "sliding_window"

    def evaluate(self, telemetry_window: List[np.ndarray]) -> WorkflowState:
        if len(telemetry_window) < self.window_size:
            return WorkflowState.ACTIVE

        window = telemetry_window[-self.window_size:]
        losses = np.array([float(v[self.loss_index]) for v in window])
        mean_loss = np.mean(losses)

        if mean_loss >= self.loss_threshold:
            return WorkflowState.ACTIVE

        # Check improvement rate: compare first half vs second half
        half = self.window_size // 2
        first_half_mean = np.mean(losses[:half])
        second_half_mean = np.mean(losses[half:])
        improvement = first_half_mean - second_half_mean

        if improvement < self.min_improvement:
            return WorkflowState.DORMANT
        return WorkflowState.ACTIVE


class CompositePolicy(Policy):
    """Freeze when multiple signals agree that AI is no longer helping.

    Evaluates three independent signals from the telemetry vector:
      [0] = normalized training loss (relative to first loss observed)
      [1] = mean simulation RMSD (application quality)
      [2] = inference stability (0-1, fraction of restart points unchanged)

    Freeze when:
      - Training loss has plateaued (relative improvement < threshold), AND
      - RMSD is not improving (no downward trend in recent window), OR
      - Inference output is stable (>stability_threshold of restarts unchanged)

    Resume when RMSD starts increasing (distribution shift).

    Telemetry vector layout:
      v = [normalized_loss, mean_rmsd, inference_stability]
    """

    LOSS_IDX = 0
    RMSD_IDX = 1
    STABILITY_IDX = 2

    def __init__(
        self,
        window_size: int = 5,
        loss_plateau_threshold: float = 0.02,
        rmsd_improvement_threshold: float = 0.1,
        stability_threshold: float = 0.8,
    ):
        self.window_size = window_size
        self.loss_plateau_threshold = loss_plateau_threshold
        self.rmsd_improvement_threshold = rmsd_improvement_threshold
        self.stability_threshold = stability_threshold

    @property
    def name(self) -> str:
        return "composite"

    def _has_loss_plateaued(self, losses: np.ndarray) -> bool:
        """Check if normalized loss has stopped improving."""
        if len(losses) < 2:
            return False
        half = len(losses) // 2
        first_half = np.mean(losses[:half])
        second_half = np.mean(losses[half:])
        if first_half == 0:
            return True
        relative_improvement = (first_half - second_half) / abs(first_half)
        return relative_improvement < self.loss_plateau_threshold

    def _is_rmsd_improving(self, rmsds: np.ndarray) -> bool:
        """Check if RMSD is trending downward (science improving)."""
        if len(rmsds) < 2:
            return True  # Assume improving until we have data
        half = len(rmsds) // 2
        first_half = np.mean(rmsds[:half])
        second_half = np.mean(rmsds[half:])
        improvement = first_half - second_half
        return improvement > self.rmsd_improvement_threshold

    def _is_rmsd_degrading(self, rmsds: np.ndarray) -> bool:
        """Check if RMSD is trending upward (distribution shift)."""
        if len(rmsds) < 2:
            return False
        half = len(rmsds) // 2
        first_half = np.mean(rmsds[:half])
        second_half = np.mean(rmsds[half:])
        degradation = second_half - first_half
        return degradation > self.rmsd_improvement_threshold

    def _is_inference_stable(self, stabilities: np.ndarray) -> bool:
        """Check if inference output has stabilized."""
        if len(stabilities) == 0:
            return False
        return np.mean(stabilities[-3:]) > self.stability_threshold

    def evaluate(self, telemetry_window: List[np.ndarray]) -> WorkflowState:
        if len(telemetry_window) < self.window_size:
            return WorkflowState.ACTIVE

        window = telemetry_window[-self.window_size:]
        losses = np.array([float(v[self.LOSS_IDX]) for v in window])
        rmsds = np.array([float(v[self.RMSD_IDX]) for v in window])
        stabilities = np.array([float(v[self.STABILITY_IDX]) for v in window])

        loss_plateaued = self._has_loss_plateaued(losses)
        rmsd_improving = self._is_rmsd_improving(rmsds)
        rmsd_degrading = self._is_rmsd_degrading(rmsds)
        inference_stable = self._is_inference_stable(stabilities)

        # RESUME: if RMSD is degrading, the model needs retraining
        if rmsd_degrading:
            return WorkflowState.RESUME

        # DORMANT: loss plateaued AND (RMSD not improving OR inference stable)
        if loss_plateaued and (not rmsd_improving or inference_stable):
            return WorkflowState.DORMANT

        return WorkflowState.ACTIVE


class MannKendallPolicy(Policy):
    """O(Nk^2) — freeze when a Mann-Kendall trend test detects no
    significant downward trend in the loss over the last k vectors.

    Uses the classical non-parametric test: if the loss is no longer
    trending downward, training has converged and can be frozen.
    """

    def __init__(
        self,
        window_size: int = 20,
        significance_level: float = 0.05,
        loss_index: int = 0,
    ):
        self.window_size = window_size
        self.significance_level = significance_level
        self.loss_index = loss_index

    @property
    def name(self) -> str:
        return "mann_kendall"

    @staticmethod
    def _mann_kendall_s(data: np.ndarray) -> float:
        """Compute the Mann-Kendall S statistic.

        S = sum_{i<j} sign(x_j - x_i)
        Positive S → upward trend, negative S → downward trend.
        """
        n = len(data)
        s = 0.0
        for i in range(n - 1):
            for j in range(i + 1, n):
                diff = data[j] - data[i]
                if diff > 0:
                    s += 1
                elif diff < 0:
                    s -= 1
        return s

    @staticmethod
    def _mann_kendall_variance(n: int) -> float:
        """Variance of S under the null hypothesis (no trend)."""
        return n * (n - 1) * (2 * n + 5) / 18.0

    def evaluate(self, telemetry_window: List[np.ndarray]) -> WorkflowState:
        if len(telemetry_window) < self.window_size:
            return WorkflowState.ACTIVE

        window = telemetry_window[-self.window_size:]
        losses = np.array([float(v[self.loss_index]) for v in window])

        s = self._mann_kendall_s(losses)
        var_s = self._mann_kendall_variance(len(losses))
        std_s = np.sqrt(var_s) if var_s > 0 else 1.0

        # Normalized test statistic
        if s > 0:
            z = (s - 1) / std_s
        elif s < 0:
            z = (s + 1) / std_s
        else:
            z = 0.0

        # Two-sided critical value approximation for common significance levels
        z_crit = {0.10: 1.645, 0.05: 1.96, 0.01: 2.576}.get(
            self.significance_level, 1.96
        )

        # If z is NOT significantly negative, there's no downward trend → freeze
        if z > -z_crit:
            return WorkflowState.DORMANT
        return WorkflowState.ACTIVE

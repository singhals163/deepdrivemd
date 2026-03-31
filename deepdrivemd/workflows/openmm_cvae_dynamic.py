"""Dynamic provisioning workflow for DeepDriveMD.

Extends DeepDriveMD_OpenMM_CVAE with signal-driven freeze/resume using:
- Signal Monitor: policy-based decision on AI convergence
- Stateful Service: Redis-backed model checkpoint management
- Resource Broker: GPU reallocation with cooldown and graceful draining

This replaces the static `freeze_after_n_trains` counter with dynamic,
telemetry-driven decisions.
"""
import logging
import time
from argparse import ArgumentParser
from functools import partial, update_wrapper
from pathlib import Path
from queue import Queue
from typing import Any, Optional

import numpy as np
from colmena.queue.python import PipeQueues
from colmena.task_server import ParslTaskServer
from proxystore.store import register_store
from proxystore.store.file import FileStore

from deepdrivemd.api import (
    DeepDriveMDSettings,
    DeepDriveMDWorkflow,
    SimulationCountDoneCallback,
    TimeoutDoneCallback,
)
from deepdrivemd.apps.cvae_inference import (
    CVAEInferenceInput,
    CVAEInferenceOutput,
    CVAEInferenceSettings,
)
from deepdrivemd.apps.cvae_train import (
    CVAETrainInput,
    CVAETrainOutput,
    CVAETrainSettings,
)
from deepdrivemd.apps.openmm_simulation import (
    MDSimulationInput,
    MDSimulationOutput,
    MDSimulationSettings,
)
from deepdrivemd.parsl import ComputeSettingsTypes, WorkstationSettings
from deepdrivemd.resource_broker.broker import GPUState, ResourceBroker
from deepdrivemd.signal_monitor.monitor import SignalMonitor
from deepdrivemd.signal_monitor.policy import Policy, WorkflowState
from deepdrivemd.stateful_service.service import StatefulService


def run_simulation(
    input_data: MDSimulationInput, config: MDSimulationSettings
) -> MDSimulationOutput:
    from deepdrivemd.apps.openmm_simulation.app import MDSimulationApplication

    app = MDSimulationApplication(config)
    output_data = app.run(input_data)
    return output_data


def run_train(input_data: CVAETrainInput, config: CVAETrainSettings) -> CVAETrainOutput:
    from deepdrivemd.apps.cvae_train.app import CVAETrainApplication

    app = CVAETrainApplication(config)
    output_data = app.run(input_data)
    return output_data


def run_inference(
    input_data: CVAEInferenceInput, config: CVAEInferenceSettings
) -> CVAEInferenceOutput:
    from deepdrivemd.apps.cvae_inference.app import CVAEInferenceApplication

    app = CVAEInferenceApplication(config)
    output_data = app.run(input_data)
    return output_data


def run_simulation_on_ml(
    input_data: MDSimulationInput, config: MDSimulationSettings
) -> MDSimulationOutput:
    """Run simulation on the ML executor (GPU 7) after model freeze."""
    from deepdrivemd.apps.openmm_simulation.app import MDSimulationApplication

    app = MDSimulationApplication(config)
    output_data = app.run(input_data)
    return output_data


class DeepDriveMD_Dynamic(DeepDriveMDWorkflow):
    """Dynamic provisioning workflow with signal-driven freeze/resume.

    Integrates Signal Monitor, Stateful Service, and Resource Broker
    into the thinker thread to dynamically manage AI resources.
    """

    def __init__(
        self,
        simulations_per_train: int,
        simulations_per_inference: int,
        policy: Policy,
        redis_host: str = "127.0.0.1",
        redis_port: int = 6379,
        cooldown_sec: float = 30.0,
        signal_window_size: int = 50,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.model_weights_available: bool = False
        self.simulations_per_train = simulations_per_train
        self.train_input = CVAETrainInput(contact_map_paths=[], rmsd_paths=[])
        self.simulations_per_inference = simulations_per_inference
        self.inference_input = CVAEInferenceInput(
            contact_map_paths=[], rmsd_paths=[], model_weight_path=Path()
        )

        self.current_model_weight_path: Optional[Path] = None
        self.train_count = 0
        self.model_frozen = False
        self.simulation_input_queue: Queue[MDSimulationInput] = Queue()
        self._ml_gpu_sim_active = False

        # Telemetry tracking for composite signal
        self._first_loss: Optional[float] = None  # For loss normalization
        self._recent_rmsds: list = []  # Rolling RMSD from simulations
        self._last_inference_dirs: set = set()  # For inference stability
        self._inference_stability: float = 0.0  # 0-1, fraction unchanged

        # Dynamic provisioning components (all in thinker thread)
        self.signal_monitor = SignalMonitor(
            policy=policy,
            window_size=signal_window_size,
            on_state_change=self._handle_state_transition,
            cooldown_sec=0.0,  # Cooldown handled by Resource Broker
        )
        self.stateful_service = StatefulService(
            redis_host=redis_host,
            redis_port=redis_port,
        )
        self.resource_broker = ResourceBroker(
            cooldown_sec=cooldown_sec,
            on_freeze=self._on_gpu_frozen,
            on_resume=self._on_gpu_resumed,
        )

        self.logger.info(
            f"Dynamic provisioning enabled: policy={policy.name}, "
            f"cooldown={cooldown_sec}s, window={signal_window_size}"
        )

    def _handle_state_transition(
        self, old_state: WorkflowState, new_state: WorkflowState
    ) -> None:
        """Signal Monitor callback: translate workflow state to GPU actions."""
        if new_state == WorkflowState.DORMANT:
            self.logger.info(
                f"Signal Monitor: {old_state.name} -> DORMANT "
                f"(train_count={self.train_count})"
            )
            # Serialize model to Redis before freezing
            if self.current_model_weight_path is not None:
                self.stateful_service.serialize(self.current_model_weight_path)
            # Request GPU reclaim
            self.resource_broker.freeze(workflow=self)
            self.model_frozen = True

        elif new_state == WorkflowState.RESUME:
            self.logger.info(
                f"Signal Monitor: {old_state.name} -> RESUME "
                f"(distribution shift detected)"
            )
            # Restore GPU for ML work
            self.resource_broker.resume(workflow=self)
            self.model_frozen = False

    def _on_gpu_frozen(self, record: dict) -> None:
        """Resource Broker callback: GPU successfully reclaimed."""
        self.logger.info(
            f"GPU reclaimed for simulations "
            f"(drain_time={record.get('drain_time_s', 'N/A')})"
        )

    def _on_gpu_resumed(self, record: dict) -> None:
        """Resource Broker callback: GPU restored for ML."""
        self.logger.info("GPU restored for ML training/inference")
        # Deserialize model from Redis for warm-start
        checkpoint = self.stateful_service.deserialize()
        if checkpoint is not None:
            self.logger.info("Model checkpoint restored from Redis")

    def _get_sim_input(self) -> MDSimulationInput:
        """Get the next simulation input (AI-steered or initial)."""
        with self.simulation_govenor:
            if not self.simulation_input_queue.empty():
                return self.simulation_input_queue.get()
            else:
                return MDSimulationInput(sim_dir=next(self.simulation_input_dirs))

    def simulate(self) -> None:
        """Submit simulation. When model is frozen, alternate between
        htex_sim and htex_ml to keep all GPUs utilized."""
        if self.model_frozen:
            # Round-robin: every 8th call goes to ML GPU
            if self.task_counter["simulation"] % 8 == 7:
                self.simulate_on_ml()
                return
        inputs = self._get_sim_input()
        self.submit_task("simulation", inputs)

    def simulate_on_ml(self) -> None:
        """Submit a simulation to htex_ml (GPU 7) after model freeze."""
        if self.done.is_set():
            return
        inputs = self._get_sim_input()
        self.queues.send_inputs(
            inputs, method="run_simulation_on_ml", topic="simulation",
            keep_inputs=False,
        )
        self.task_counter["simulation"] += 1

    def train(self) -> None:
        self.resource_broker.register_ml_task_start()
        self.submit_task("train", self.train_input)

    def inference(self) -> None:
        while not self.model_weights_available:
            time.sleep(1)
        self.resource_broker.register_ml_task_start()
        self.submit_task("inference", self.inference_input)

    def handle_simulation_output(self, output: MDSimulationOutput) -> None:
        self.train_input.append(output.contact_map_path, output.rmsd_path)
        self.inference_input.append(output.contact_map_path, output.rmsd_path)
        num_sims = len(self.train_input)

        # Track RMSD for composite signal
        try:
            rmsd_data = np.load(output.rmsd_path)
            mean_rmsd = float(np.mean(rmsd_data))
            self._recent_rmsds.append(mean_rmsd)
            # Keep only last 50
            if len(self._recent_rmsds) > 50:
                self._recent_rmsds = self._recent_rmsds[-50:]
        except Exception:
            pass

        if not self.model_frozen and num_sims and (num_sims % self.simulations_per_train == 0):
            self.run_training.set()

        if num_sims and (num_sims % self.simulations_per_inference == 0):
            self.run_inference.set()

    def handle_train_output(self, output: CVAETrainOutput) -> None:
        self.resource_broker.register_ml_task_complete()
        self.inference_input.model_weight_path = output.model_weight_path
        self.current_model_weight_path = output.model_weight_path
        self.model_weights_available = True
        self.train_count += 1

        self.logger.info(
            f"Training cycle {self.train_count} complete, "
            f"model_weight_path: {output.model_weight_path}"
        )

        # Build composite telemetry vector: [normalized_loss, mean_rmsd, inference_stability]
        raw_loss = output.final_loss
        if self._first_loss is None:
            self._first_loss = raw_loss if raw_loss > 0 else 1.0
        normalized_loss = raw_loss / self._first_loss

        mean_rmsd = float(np.mean(self._recent_rmsds[-10:])) if self._recent_rmsds else 10.0

        metrics = np.array([normalized_loss, mean_rmsd, self._inference_stability])
        state = self.signal_monitor.submit_telemetry(metrics)
        self.logger.info(
            f"Signal Monitor: norm_loss={normalized_loss:.4f}, "
            f"rmsd={mean_rmsd:.3f}, inf_stab={self._inference_stability:.3f}, "
            f"state={state.name}, window={self.signal_monitor.get_stats()['window_size']}"
        )

    def handle_inference_output(self, output: CVAEInferenceOutput) -> None:
        self.resource_broker.register_ml_task_complete()

        # Track inference stability: what fraction of restart dirs are the same
        current_dirs = set(str(d) for d in output.sim_dirs)
        if self._last_inference_dirs:
            overlap = len(current_dirs & self._last_inference_dirs)
            total = max(len(current_dirs), 1)
            self._inference_stability = overlap / total
        self._last_inference_dirs = current_dirs

        with self.simulation_govenor:
            self.simulation_input_queue.queue.clear()
            for sim_dir, sim_frame in zip(output.sim_dirs, output.sim_frames):
                self.simulation_input_queue.put(
                    MDSimulationInput(sim_dir=sim_dir, sim_frame=sim_frame)
                )
        self.logger.info(
            f"Processed inference result and added {len(output.sim_dirs)} "
            f"new restart points (stability={self._inference_stability:.3f})."
        )

    def get_dynamic_stats(self) -> dict:
        """Collect statistics from all dynamic provisioning components."""
        return {
            "signal_monitor": self.signal_monitor.get_stats(),
            "stateful_service": self.stateful_service.get_stats(),
            "resource_broker": self.resource_broker.get_stats(),
            "train_count": self.train_count,
            "model_frozen": self.model_frozen,
        }


class DynamicExperimentSettings(DeepDriveMDSettings):
    """YAML-configurable settings for dynamic provisioning experiments."""

    simulation_settings: MDSimulationSettings
    train_settings: CVAETrainSettings
    inference_settings: CVAEInferenceSettings
    compute_settings: ComputeSettingsTypes

    # Dynamic provisioning settings
    policy_name: str = "sliding_window"
    policy_window_size: int = 10
    policy_loss_threshold: float = 0.05
    policy_min_improvement: float = 0.001
    cooldown_sec: float = 30.0
    signal_window_size: int = 50
    redis_host: str = "127.0.0.1"
    redis_port: int = 6379


def build_policy(cfg: DynamicExperimentSettings) -> Policy:
    """Build a policy instance from config settings."""
    from deepdrivemd.signal_monitor.policy import (
        CompositePolicy,
        MannKendallPolicy,
        SlidingWindowPolicy,
        ThresholdPolicy,
    )

    if cfg.policy_name == "threshold":
        return ThresholdPolicy(loss_threshold=cfg.policy_loss_threshold)
    elif cfg.policy_name == "sliding_window":
        return SlidingWindowPolicy(
            window_size=cfg.policy_window_size,
            loss_threshold=cfg.policy_loss_threshold,
            min_improvement=cfg.policy_min_improvement,
        )
    elif cfg.policy_name == "mann_kendall":
        return MannKendallPolicy(window_size=cfg.policy_window_size)
    elif cfg.policy_name == "composite":
        return CompositePolicy(
            window_size=cfg.policy_window_size,
            loss_plateau_threshold=cfg.policy_loss_threshold,
        )
    else:
        raise ValueError(f"Unknown policy: {cfg.policy_name}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()
    cfg = DynamicExperimentSettings.from_yaml(args.config)
    cfg.dump_yaml(cfg.run_dir / "params.yaml")
    cfg.configure_logging()

    policy = build_policy(cfg)
    logging.info(f"Using policy: {policy.name}")

    # Make the proxy store
    store = FileStore(name="file", store_dir=str(cfg.run_dir / "proxy-store"))
    register_store(store)

    queues = PipeQueues(
        serialization_method="pickle",
        topics=["simulation", "train", "inference"],
        proxystore_name="file",
        proxystore_threshold=10000,
    )

    parsl_config = cfg.compute_settings.config_factory(cfg.run_dir / "run-info")

    my_run_simulation = partial(run_simulation, config=cfg.simulation_settings)
    my_run_train = partial(run_train, config=cfg.train_settings)
    my_run_inference = partial(run_inference, config=cfg.inference_settings)
    update_wrapper(my_run_simulation, run_simulation)
    update_wrapper(my_run_train, run_train)
    update_wrapper(my_run_inference, run_inference)

    # Use per-method executor routing when ML GPUs are pinned
    use_ml_pinning = (
        isinstance(cfg.compute_settings, WorkstationSettings)
        and cfg.compute_settings.ml_accelerators is not None
    )
    if use_ml_pinning:
        my_run_sim_on_ml = partial(run_simulation_on_ml, config=cfg.simulation_settings)
        update_wrapper(my_run_sim_on_ml, run_simulation_on_ml)
        doer = ParslTaskServer(
            [my_run_simulation,
             (my_run_train, {'executors': ['htex_ml']}),
             (my_run_inference, {'executors': ['htex_ml']}),
             (my_run_sim_on_ml, {'executors': ['htex_ml']})],
            queues, parsl_config,
            default_executors=['htex_sim'],
        )
    else:
        doer = ParslTaskServer(
            [my_run_simulation, my_run_train, my_run_inference], queues, parsl_config
        )

    thinker = DeepDriveMD_Dynamic(
        queue=queues,
        result_dir=cfg.run_dir / "result",
        simulation_input_dir=cfg.simulation_input_dir,
        num_workers=cfg.num_workers,
        simulations_per_train=cfg.simulations_per_train,
        simulations_per_inference=cfg.simulations_per_inference,
        policy=policy,
        redis_host=cfg.redis_host,
        redis_port=cfg.redis_port,
        cooldown_sec=cfg.cooldown_sec,
        signal_window_size=cfg.signal_window_size,
        done_callbacks=[
            SimulationCountDoneCallback(cfg.num_total_simulations),
            TimeoutDoneCallback(cfg.duration_sec),
        ],
    )
    logging.info("Created dynamic provisioning task server and thinker")

    try:
        doer.start()
        thinker.start()
        logging.info("Launched the servers")
        thinker.join()
        logging.info("Task generator has completed")
    finally:
        queues.send_kill_signal()

        # Dump dynamic provisioning statistics
        import json
        stats = thinker.get_dynamic_stats()
        stats_path = cfg.run_dir / "dynamic_provisioning_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2, default=str)
        logging.info(f"Dynamic provisioning stats saved to {stats_path}")

    doer.join()
    store.close()

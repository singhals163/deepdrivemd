import logging
import time
from argparse import ArgumentParser
from functools import partial, update_wrapper
from pathlib import Path
from queue import Queue
from typing import Any

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
from deepdrivemd.parsl import ComputeSettingsTypes


def run_simulation(
    input_data: MDSimulationInput, config: MDSimulationSettings
) -> MDSimulationOutput:
    from deepdrivemd.apps.openmm_simulation.app import MDSimulationApplication
    app = MDSimulationApplication(config)
    return app.run(input_data)

# =============================================================================
# STATEFUL TRAINING WRAPPERS
# =============================================================================
def run_train(
    ai_state: int, model_ref: Any, input_data: CVAETrainInput, config: CVAETrainSettings
) -> CVAETrainOutput:
    """Wrapper that resolves the state of the AI model before training."""
    from deepdrivemd.apps.cvae_train.app import CVAETrainApplication
    app = CVAETrainApplication(config)
    
    if ai_state == 0:
        # Cold Start: Init model -> Store in GPU Cache -> Train
        new_ref = app.startup()
        output_data = app.run_active(new_ref, input_data)
        output_data.model_ref = new_ref  # Piggyback reference back to Thinker
        return output_data
        
    elif ai_state == 1:
        # Wakeup: Deserialize Proxy -> Load to GPU Cache -> Train
        new_ref = app.wakeup(model_ref) 
        output_data = app.run_active(new_ref, input_data)
        output_data.model_ref = new_ref
        return output_data
        
    else:
        # Active: Train directly against cached GPU memory
        output_data = app.run_active(model_ref, input_data)
        output_data.model_ref = model_ref
        return output_data

def run_admin(model_ref: str, config: CVAETrainSettings) -> Any:
    """Executes the State 2 -> State 1 transition (Sleep and Serialize)."""
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


class DeepDriveMD_OpenMM_CVAE(DeepDriveMDWorkflow):
    def __init__(
        self, simulations_per_train: int, simulations_per_inference: int, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.model_weights_available: bool = False
        self.simulations_per_train = simulations_per_train
        self.train_input = CVAETrainInput(contact_map_paths=[], rmsd_paths=[])
        self.simulations_per_inference = simulations_per_inference
        self.inference_input = CVAEInferenceInput(
            contact_map_paths=[], rmsd_paths=[], model_weight_path=Path()
        )
        self.simulation_input_queue: Queue[MDSimulationInput] = Queue()

    def simulate(self) -> None:
        with self.simulation_govenor:
            if not self.simulation_input_queue.empty():
                inputs = self.simulation_input_queue.get()
            else:
                inputs = MDSimulationInput(sim_dir=next(self.simulation_input_dirs))
        self.submit_task("simulation", inputs)

    def train(self) -> None:
        """Dispatches the training task with the current state and corresponding reference."""
        model_ref = self.dormant_model_proxy if self.ai_state == 1 else self.active_model_ref
        
        # Topic "train" routes to run_train
        self.submit_task("train", self.ai_state, model_ref, self.train_input)
        
        # Optional: self.train_input.clear() if you implement incremental training

    def inference(self) -> None:
        while not self.model_weights_available:
            time.sleep(1)
        self.submit_task("inference", self.inference_input)

    def handle_simulation_output(self, output: MDSimulationOutput) -> None:
        self.train_input.append(output.contact_map_path, output.rmsd_path)
        self.inference_input.append(output.contact_map_path, output.rmsd_path)
        
        num_sims = len(self.train_input)

        if num_sims and (num_sims % self.simulations_per_train == 0):
            self.run_training.set()

        if num_sims and (num_sims % self.simulations_per_inference == 0):
            self.run_inference.set()

    def handle_train_output(self, output: CVAETrainOutput) -> None:
        self.inference_input.model_weight_path = output.model_weight_path
        self.model_weights_available = True
        self.logger.info(f"Updated model_weight_path to: {output.model_weight_path}")

    def handle_inference_output(self, output: CVAEInferenceOutput) -> None:
        with self.simulation_govenor:
            self.simulation_input_queue.queue.clear()
            for sim_dir, sim_frame in zip(output.sim_dirs, output.sim_frames):
                self.simulation_input_queue.put(
                    MDSimulationInput(sim_dir=sim_dir, sim_frame=sim_frame)
                )

        self.logger.info(
            f"processed inference result and added {len(output.sim_dirs)} "
            "new restart points to the simulation_input_queue."
        )


class ExperimentSettings(DeepDriveMDSettings):
    """Provide a YAML interface to configure the experiment."""

    simulation_settings: MDSimulationSettings
    train_settings: CVAETrainSettings
    inference_settings: CVAEInferenceSettings
    compute_settings: ComputeSettingsTypes


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-t", "--test", action="store_true", help="Test Mock Application")
    args = parser.parse_args()
    
    cfg = ExperimentSettings.from_yaml(args.config)
    cfg.dump_yaml(cfg.run_dir / "params.yaml")
    cfg.configure_logging()

    store = FileStore(name="file", store_dir=str(cfg.run_dir / "proxy-store"))
    register_store(store)

    # Added "admin" topic for state 2 -> 1 serialization tasks
    queues = PipeQueues(
        serialization_method="pickle",
        topics=["simulation", "train", "inference", "admin"],
        proxystore_name="file",
        proxystore_threshold=10000,
    )

    parsl_config = cfg.compute_settings.config_factory(cfg.run_dir / "run-info")

    my_run_simulation = partial(run_simulation, config=cfg.simulation_settings)
    my_run_train = partial(run_train, config=cfg.train_settings)
    my_run_admin = partial(run_admin, config=cfg.train_settings)
    my_run_inference = partial(run_inference, config=cfg.inference_settings)
    
    update_wrapper(my_run_simulation, run_simulation)
    update_wrapper(my_run_train, run_train)
    update_wrapper(my_run_admin, run_admin)
    update_wrapper(my_run_inference, run_inference)

    parsl_executors = {exec.label: exec for exec in parsl_config.executors}

    doer = ParslTaskServer(
        [my_run_simulation, my_run_train, my_run_inference, my_run_admin], queues, parsl_config
    )

    thinker = DeepDriveMD_OpenMM_CVAE(
        queue=queues,
        result_dir=cfg.run_dir / "result",
        simulation_input_dir=cfg.simulation_input_dir,
        num_workers=cfg.num_workers,
        simulations_per_train=cfg.simulations_per_train,
        simulations_per_inference=cfg.simulations_per_inference,
        done_callbacks=[
            SimulationCountDoneCallback(cfg.num_total_simulations),
            TimeoutDoneCallback(cfg.duration_sec),
        ],
    )
    logging.info("Created the task server and task generator")

    try:
        doer.start()
        thinker.start()
        logging.info("Launched the servers")
        thinker.join()
        logging.info("Task generator has completed")
    finally:
        queues.send_kill_signal()

    doer.join()
    store.close()
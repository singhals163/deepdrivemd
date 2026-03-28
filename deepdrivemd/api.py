import itertools
import json
import logging
import shutil
import sys
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from threading import Event, Semaphore
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import yaml
from colmena.models import Result
from colmena.queue import ColmenaQueues
from colmena.thinker import BaseThinker, agent, event_responder, result_processor
from pydantic import BaseSettings as _BaseSettings
from pydantic import root_validator, validator

_T = TypeVar("_T")

PathLike = Union[str, Path]


def _resolve_path_exists(value: Optional[Path]) -> Optional[Path]:
    if value is None:
        return None
    p = value.resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def path_validator(field: str) -> classmethod:
    decorator = validator(field, allow_reuse=True)
    _validator = decorator(_resolve_path_exists)
    return _validator


class BaseSettings(_BaseSettings):
    def dump_yaml(self, filename: PathLike) -> None:
        with open(filename, mode="w") as fp:
            yaml.dump(json.loads(self.json()), fp, indent=4, sort_keys=False)

    @classmethod
    def from_yaml(cls: Type[_T], filename: PathLike) -> _T:
        with open(filename) as fp:
            raw_data = yaml.safe_load(fp)
        return cls(**raw_data)


class ApplicationSettings(BaseSettings):
    output_dir: Path
    node_local_path: Optional[Path] = None

    @validator("output_dir")
    def create_output_dir(cls, v: Path) -> Path:
        v = v.resolve()
        v.mkdir(exist_ok=True, parents=True)
        return v


class BatchSettings(BaseSettings):
    def __len__(self) -> int:
        lists = self.get_lists()
        return len(lists[0]) if lists else 0

    def get_lists(self) -> List[List[Any]]:
        return [field for field in self.__dict__.values() if isinstance(field, list)]

    def append(self, *args: Any) -> None:
        lists = self.get_lists()
        assert len(lists) == len(args), "Number of args must match the number of lists."
        for arg, _list in zip(args, lists):
            _list.append(arg)

    def clear(self) -> None:
        for _list in self.get_lists():
            _list.clear()


class DeepDriveMDSettings(BaseSettings):
    experiment_name: str = "experiment"
    runs_dir: Path = Path("runs")
    run_dir: Path
    redishost: str = "127.0.0.1"
    redisport: int = 6379
    simulation_input_dir: Path
    num_total_simulations: int
    duration_sec: float = float("inf")
    num_workers: int
    simulations_per_train: int
    simulations_per_inference: int

    simulation_settings: ApplicationSettings
    train_settings: ApplicationSettings
    inference_settings: ApplicationSettings

    def configure_logging(self) -> None:
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=logging.INFO,
            handlers=[
                logging.FileHandler(self.run_dir / "runtime.log"),
                logging.StreamHandler(sys.stdout),
            ],
        )

    @root_validator(pre=True)
    def create_output_dirs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        runs_dir = Path(values.get("runs_dir", "runs")).resolve()
        experiment_name = values.get("experiment_name", "experiment")
        timestamp = datetime.now().strftime("%d%m%y-%H%M%S")
        run_dir = runs_dir / f"{experiment_name}-{timestamp}"
        run_dir.mkdir(exist_ok=False, parents=True)
        values["run_dir"] = run_dir
        for name in ["simulation", "train", "inference"]:
            values[f"{name}_settings"]["output_dir"] = run_dir / name
        return values

    _simulation_input_dir_exists = path_validator("simulation_input_dir")


class Application(ABC):
    def __init__(self, config: ApplicationSettings) -> None:
        self.config = config
        self.__workdir: Optional[Path] = None

    @property
    def persistent_dir(self) -> Path:
        return self.config.output_dir / self.workdir.name

    @property
    def workdir(self) -> Path:
        if isinstance(self.__workdir, Path):
            return self.__workdir

        workdir_parent = (
            self.config.output_dir
            if self.config.node_local_path is None
            else self.config.node_local_path
        )
        timestamp = datetime.now().strftime("%d%m%y-%H%M%S")
        workdir = workdir_parent / f"run-{timestamp}-{uuid.uuid4()}"
        workdir.mkdir(exist_ok=True, parents=True)
        self.__workdir = workdir
        return workdir

    def backup_node_local(self) -> None:
        if self.config.node_local_path is not None:
            if next(self.workdir.iterdir(), None):
                shutil.move(self.workdir, self.persistent_dir)

    def copy_to_workdir(self, p: Path) -> Path:
        if p.is_file():
            return Path(shutil.copy(p, self.workdir))
        else:
            return Path(shutil.copytree(p, self.workdir / p.name))


class DoneCallback(ABC):
    @abstractmethod
    def workflow_finished(self, workflow: "DeepDriveMDWorkflow") -> bool:
        ...


class TimeoutDoneCallback(DoneCallback):
    def __init__(self, duration_sec: float) -> None:
        self.duration_sec = duration_sec
        self.start_time = time.time()

    def workflow_finished(self, workflow: "DeepDriveMDWorkflow") -> bool:
        elapsed_sec = time.time() - self.start_time
        return elapsed_sec > self.duration_sec


class SimulationCountDoneCallback(DoneCallback):
    def __init__(self, total_simulations: int) -> None:
        self.total_simulations = total_simulations

    def workflow_finished(self, workflow: "DeepDriveMDWorkflow") -> bool:
        return workflow.task_counter["simulation"] >= self.total_simulations


class InferenceCountDoneCallback(DoneCallback):
    def __init__(self, total_inferences: int) -> None:
        self.total_inferences = total_inferences

    def workflow_finished(self, workflow: "DeepDriveMDWorkflow") -> bool:
        return workflow.task_counter["inference"] >= self.total_inferences


class DeepDriveMDWorkflow(BaseThinker):  # type: ignore[misc]
    def __init__(
        self,
        queue: ColmenaQueues,
        result_dir: Path,
        simulation_input_dir: Path,
        num_workers: int,
        done_callbacks: List[DoneCallback],
    ) -> None:
        super().__init__(queue)

        result_dir.mkdir(exist_ok=True)
        self.result_dir = result_dir
        self.num_workers = num_workers

        self.simulation_input_dirs = itertools.cycle(
            filter(lambda p: p.is_dir(), simulation_input_dir.glob("*"))
        )

        self.simulations_completed = 0
        self.task_counter: defaultdict[str, int] = defaultdict(int)
        self.done_callbacks = done_callbacks

        self.simulation_govenor = Semaphore()
        self.run_training = Event()
        self.run_inference = Event()

    def log_result(self, result: Result, topic: str) -> None:
        """Write a JSON result per line of the output file."""
        with open(self.result_dir / f"{topic}.json", "a") as f:
            print(result.json(exclude={"inputs", "value"}), file=f)

    def submit_task(self, topic: str, *inputs: Any) -> None:
        self.queues.send_inputs(
            *inputs, method=f"run_{topic}", topic=topic, keep_inputs=False
        )
        self.task_counter[topic] += 1

    @agent
    def main_loop(self) -> None:
        while not self.done.is_set():
            for callback in self.done_callbacks:
                if callback.workflow_finished(self):
                    self.logger.info("Exiting DeepDriveMD")
                    self.done.set()
                    return
            time.sleep(1)

    @agent(startup=True)
    def start_simulations(self) -> None:
        for _ in range(self.num_workers - 1):
            self.simulate()

    @result_processor(topic="simulation")  # type: ignore[misc]
    def process_simulation_result(self, result: Result) -> None:
        self.log_result(result, "simulation")
        if not result.success:
            return self.logger.warning("Bad simulation result")
            
        self.simulations_completed += 1

        # Non-blocking telemetry ingestion
        self.telemetry_queue.put(("simulation", result.value))

        if self.done.is_set():
            return

        self.simulate()
        self.handle_simulation_output(result.value)

    @event_responder(event_name="run_training")
    def perform_training(self) -> None:
        self.logger.info("Started training process")
        self.train()

        result: Result = self.queues.get_result(topic="train")
        self.logger.info("Received training result")
        self.log_result(result, "train")
        
        if not result.success:
            return self.logger.warning("Bad train result")

        # Process the training output
        self.handle_train_output(result.value)
        self.logger.info("Training process is complete")

    # TODO (wardlt): We can have this event_responder allocate resources away from simulation if desired.
    @event_responder(event_name="run_inference")  # type: ignore[misc]
    def perform_inference(self) -> None:
        self.logger.info("Started inference process")
        self.inference()

        result: Result = self.queues.get_result(topic="inference")
        self.logger.info("Received inference result")

        self.log_result(result, "inference")
        if not result.success:
            return self.logger.warning("Bad inference result")

        self.handle_inference_output(result.value)
        self.logger.info("Inference process is complete")

    @abstractmethod
    def simulate(self) -> None:
        ...

    @abstractmethod
    def train(self) -> None:
        ...

    @abstractmethod
    def inference(self) -> None:
        ...

    @abstractmethod
    def handle_simulation_output(self, output: Any) -> None:
        ...

    @abstractmethod
    def handle_train_output(self, output: Any) -> None:
        ...

    @abstractmethod
    def handle_inference_output(self, output: Any) -> None:
        ...
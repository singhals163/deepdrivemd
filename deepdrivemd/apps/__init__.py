import time
import json
import fcntl  # Standard library on Linux/macOS for file locking
from pathlib import Path

class SimpleProfiler:
    def __init__(self, log_file:Path, name="unknown_task"):
        """
        name: Identifier for this run (e.g., 'training_run_1', 'sim_run_5')
        log_file: The common file path to write results to.
        """
        self.stats = {}
        self.name = name
        # create a file in parent(parent(log_file))) by the name "profile_stats.jsonl" if it does not exist, otherwise use it
        self.log_file = Path(log_file).parent.parent / "profile_stats.jsonl"

    def __call__(self, label):
        parent = self
        class Timer:
            def __enter__(self):
                self.start = time.perf_counter()
            def __exit__(self, *args):
                elapsed = time.perf_counter() - self.start
                # Store directly in parent's stats
                parent.stats[label] = elapsed
        return Timer()

    def save(self):
        """Writes the collected stats to the common log file safely."""
        if not self.stats:
            return

        # Structure the log entry
        data = {
            "timestamp": time.time(),
            "task_type": self.name,
            "metrics": self.stats
        }

        # Safe append with locking
        try:
            with open(self.log_file, "a") as f:
                # Acquire an exclusive lock to prevent write collisions from other workers
                fcntl.flock(f, fcntl.LOCK_EX) 
                f.write(json.dumps(data) + "\n")
                # Unlock happens automatically when file is closed, but good practice to release
                fcntl.flock(f, fcntl.LOCK_UN)
        except Exception as e:
            print(f"Profiler failed to write to {self.log_file}: {e}")

    def __del__(self):
        """Automatically save when the profiler instance is destroyed."""
        self.save()
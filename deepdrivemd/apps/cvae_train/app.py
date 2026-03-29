import numpy as np
import pandas as pd
import torch
import gc
from pathlib import Path
from typing import Any, Dict
from mdlearn.nn.models.vae.symmetric_conv2d_vae import SymmetricConv2dVAETrainer
from natsort import natsorted

from deepdrivemd.api import Application
from deepdrivemd.apps.cvae_train import (
    CVAESettings,
    CVAETrainInput,
    CVAETrainOutput,
    CVAETrainSettings,
)

# =============================================================================
# WORKER-LEVEL GLOBAL CACHE
# Holds the active trainer in GPU memory between Parsl task executions.
# =============================================================================
_GPU_MODEL_CACHE: Dict[str, SymmetricConv2dVAETrainer] = {}


class CVAETrainApplication(Application):
    """Refactored Application to handle stateful execution steps without I/O bloat."""
    config: CVAETrainSettings

    def _init_trainer(self) -> SymmetricConv2dVAETrainer:
        """Helper to initialize a fresh trainer."""
        cvae_settings = CVAESettings.from_yaml(self.config.cvae_settings_yaml).dict()
        return SymmetricConv2dVAETrainer(**cvae_settings)

    def startup(self) -> str:
        """State 0 -> 2: Initialize model and load into GPU cache."""
        trainer = self._init_trainer()

        if self.config.checkpoint_path is not None:
            checkpoint = torch.load(self.config.checkpoint_path, map_location=trainer.device)
            trainer.model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint and trainer.optimizer:
                trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        model_ref = "active_cvae_trainer"
        _GPU_MODEL_CACHE[model_ref] = trainer
        return model_ref

    def run_active(self, model_ref: str, input_data: CVAETrainInput) -> CVAETrainOutput:
        """State 2: Execute training using the cached GPU model."""
        if model_ref not in _GPU_MODEL_CACHE:
            raise RuntimeError(f"Model ref '{model_ref}' not found in worker cache. Startup required.")

        trainer = _GPU_MODEL_CACHE[model_ref]

        # Load data directly into memory
        contact_maps = np.concatenate([np.load(p, allow_pickle=True) for p in input_data.contact_map_paths])
        rmsds = np.concatenate([np.load(p) for p in input_data.rmsd_paths])

        # Train model
        model_dir = self.workdir / "model"
        # Note: trainer.fit may still write its internal epoch checkpoints depending on config
        trainer.fit(X=contact_maps, scalars={"rmsd": rmsds}, output_path=model_dir)

        # Extract telemetry directly from memory
        loss_df = pd.DataFrame(trainer.loss_curve_)
        final_train_loss = float(loss_df["train_loss"].iloc[-1]) if "train_loss" in loss_df else 0.0
        final_valid_loss = float(loss_df["valid_loss"].iloc[-1]) if "valid_loss" in loss_df else 0.0
        
        telemetry = {
            "validation_loss": final_valid_loss,
            "training_loss": final_train_loss,
        }

        # Locate the checkpoint for inference tasks
        checkpoint_dir = model_dir / "checkpoints"
        model_weight_path = natsorted(list(checkpoint_dir.glob("*.pt")))[-1]
        
        # If backup_node_local() is bypassed to save I/O time, inference MUST 
        # read from the original workdir, not the persistent_dir.
        # model_weight_path = self.persistent_dir / "model" / "checkpoints" / model_weight_path.name

        output_data = CVAETrainOutput(
            model_weight_path=model_weight_path, 
            telemetry=telemetry
        )

        return output_data

    def transition_to_dormant(self, model_ref: str) -> Dict[str, Any]:
        """State 2 -> 1: Extract state, clear GPU memory, return bytes for ProxyStore."""
        trainer = _GPU_MODEL_CACHE.pop(model_ref, None)
        if not trainer:
            raise RuntimeError("Cannot transition to dormant; model not active.")
        
        state = {
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict() if trainer.optimizer else None
        }

        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        
        return state

    def wakeup(self, state_dict_proxy: Dict[str, Any]) -> str:
        """State 1 -> 2: Re-initialize and load state from ProxyStore."""
        trainer = self._init_trainer()
        
        trainer.model.load_state_dict(state_dict_proxy["model_state_dict"])
        if state_dict_proxy.get("optimizer_state_dict") and trainer.optimizer:
            trainer.optimizer.load_state_dict(state_dict_proxy["optimizer_state_dict"])
            
        model_ref = "active_cvae_trainer"
        _GPU_MODEL_CACHE[model_ref] = trainer
        return model_ref

    def transition_to_terminated(self, model_ref: str) -> Path:
        """State 2 -> 0: Extract final state, dump to persistent storage, clear GPU."""
        trainer = _GPU_MODEL_CACHE.pop(model_ref, None)
        if not trainer:
            raise RuntimeError("Cannot terminate; model not active.")

        # Define the final checkpoint dump path
        final_checkpoint_path = self.persistent_dir / "final_model_checkpoint.pt"
        final_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Execute the single I/O operation to save the permanent model
        torch.save({
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict() if trainer.optimizer else None
        }, final_checkpoint_path)

        # Wipe the GPU
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        
        return final_checkpoint_path

    def terminate_from_dormant(self, state_dict_proxy: Dict[str, Any]) -> Path:
        """State 1 -> 0: Read from ProxyStore, dump to disk, return path."""
        final_checkpoint_path = self.persistent_dir / "final_model_checkpoint.pt"
        final_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # ProxyStore automatically resolves the proxy into the actual dictionary
        # the moment torch.save tries to access it.
        torch.save(state_dict_proxy, final_checkpoint_path)

        return final_checkpoint_path

    def init_to_dormant(self) -> Dict[str, Any]:
        """State 0 -> 1: Initialize model, immediately extract state, return for proxying."""
        trainer = self._init_trainer()

        if self.config.checkpoint_path is not None:
            checkpoint = torch.load(self.config.checkpoint_path, map_location=trainer.device)
            trainer.model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint and trainer.optimizer:
                trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        state = {
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict() if trainer.optimizer else None
        }

        # Aggressively clear the GPU before the worker returns
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

        return state
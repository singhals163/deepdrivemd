import numpy as np
import pandas as pd
import torch
import gc
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
    """Refactored Application to handle stateful execution steps."""
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
            # Load optimizer state if necessary to maintain momentum across cycles
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

        input_data.dump_yaml(self.workdir / "input.yaml")

        # Load data. 
        # *CRITICAL*: To eliminate the AI Tax compounding cost, ensure 'input_data' 
        # only contains newly generated simulation paths since the last cycle, 
        # not the full historical dataset.
        contact_maps = np.concatenate([np.load(p, allow_pickle=True) for p in input_data.contact_map_paths])
        rmsds = np.concatenate([np.load(p) for p in input_data.rmsd_paths])

        # Train model (warm started automatically because trainer is cached)
        model_dir = self.workdir / "model"
        trainer.fit(X=contact_maps, scalars={"rmsd": rmsds}, output_path=model_dir)

        # Log the loss
        loss_df = pd.DataFrame(trainer.loss_curve_)
        loss_df.to_csv(model_dir / "loss.csv")

        # --- Extract Telemetry for Signal Monitor ---
        # Get the final training and validation loss of this specific cycle
        final_train_loss = float(loss_df["train_loss"].iloc[-1]) if "train_loss" in loss_df else 0.0
        final_valid_loss = float(loss_df["valid_loss"].iloc[-1]) if "valid_loss" in loss_df else 0.0
        
        telemetry = {
            "validation_loss": final_valid_loss,
            "training_loss": final_train_loss,
        }

        # Save and locate checkpoint
        checkpoint_dir = model_dir / "checkpoints"
        model_weight_path = natsorted(list(checkpoint_dir.glob("*.pt")))[-1]
        model_weight_path = self.persistent_dir / "model" / "checkpoints" / model_weight_path.name

        output_data = CVAETrainOutput(
            model_weight_path=model_weight_path, 
            telemetry=telemetry
        )
        output_data.dump_yaml(self.workdir / "output.yaml")
        self.backup_node_local()

        return output_data

    def transition_to_dormant(self, model_ref: str) -> Dict[str, Any]:
        """State 2 -> 1: Extract state, clear GPU memory, return bytes for ProxyStore."""
        trainer = _GPU_MODEL_CACHE.pop(model_ref, None)
        if not trainer:
            raise RuntimeError("Cannot transition to dormant; model not active.")
        
        # Extract PyTorch state dictionaries
        state = {
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict() if trainer.optimizer else None
        }

        # Explicitly destroy the trainer and clear the 11.9 GB residual footprint
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
import time

import numpy as np
import pandas as pd
import torch
from mdlearn.nn.models.vae.symmetric_conv2d_vae import SymmetricConv2dVAETrainer
from natsort import natsorted

from deepdrivemd.api import Application
from deepdrivemd.apps.cvae_train import (
    CVAESettings,
    CVAETrainInput,
    CVAETrainOutput,
    CVAETrainSettings,
)


class CVAETrainApplication(Application):
    config: CVAETrainSettings

    def run(self, input_data: CVAETrainInput) -> CVAETrainOutput:
        # Log the input data
        input_data.dump_yaml(self.workdir / "input.yaml")

        # Initialize the model
        cvae_settings = CVAESettings.from_yaml(self.config.cvae_settings_yaml).dict()
        trainer = SymmetricConv2dVAETrainer(**cvae_settings)

        if self.config.checkpoint_path is not None:
            checkpoint = torch.load(
                self.config.checkpoint_path, map_location=trainer.device
            )
            trainer.model.load_state_dict(checkpoint["model_state_dict"])

        # Load contact maps. Simulations save them as object arrays of
        # sparse COO indices (concat of row, col arrays). When all frames
        # in a sim have the same number of contacts, numpy auto-promotes
        # to a 2D array which breaks concatenation with 1D object arrays
        # from other sims. Normalize to 1D object arrays.
        def _load_sparse_maps(path):
            arr = np.load(path, allow_pickle=True)
            if arr.ndim == 2:
                # numpy auto-promoted sparse COO to 2D when all frames
                # had equal contacts, changing inner dtype to object.
                # Flatten back to 1D object array with int16 elements.
                out = np.empty(len(arr), dtype=object)
                for i in range(len(arr)):
                    out[i] = np.asarray(arr[i], dtype=np.int16)
                return out
            return arr

        contact_maps = np.concatenate(
            [_load_sparse_maps(p) for p in input_data.contact_map_paths]
        )
        rmsds = np.concatenate([np.load(p) for p in input_data.rmsd_paths])

        # Train model
        model_dir = self.workdir / "model"  # Need to create new directory
        t_train_start = time.perf_counter()
        trainer.fit(X=contact_maps, scalars={"rmsd": rmsds}, output_path=model_dir)
        training_time_s = time.perf_counter() - t_train_start

        # Log the loss
        pd.DataFrame(trainer.loss_curve_).to_csv(model_dir / "loss.csv")

        # Get the most recent model checkpoint
        checkpoint_dir = model_dir / "checkpoints"
        model_weight_path = natsorted(list(checkpoint_dir.glob("*.pt")))[-1]
        # Adjust the path to the persistent path if using node local storage.
        model_weight_path = (
            self.persistent_dir / "model" / "checkpoints" / model_weight_path.name
        )

        # Extract all ML signals for the signal monitor
        lc = trainer.loss_curve_
        output_data = CVAETrainOutput(
            model_weight_path=model_weight_path,
            final_loss=float(lc["train_loss"][-1]) if lc else 0.0,
            final_valid_loss=float(lc["valid_loss"][-1]) if lc and "valid_loss" in lc else 0.0,
            final_recon_loss=float(lc["train_recon_loss"][-1]) if lc and "train_recon_loss" in lc else 0.0,
            final_kld_loss=float(lc["train_kld_loss"][-1]) if lc and "train_kld_loss" in lc else 0.0,
            training_time_s=training_time_s,
            num_training_samples=len(contact_maps),
        )
        # Log the output data
        output_data.dump_yaml(self.workdir / "output.yaml")
        self.backup_node_local()

        return output_data

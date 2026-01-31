import numpy as np
import nvtx
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
        # [PROFILE] Unique ID: Use the work directory name (usually a UUID)
        # This ensures that if multiple trainings happen, they don't overlap in the visualizer
        train_id = self.workdir.name

        # [PROFILE] Metadata I/O
        with nvtx.annotate(f"Train_Log_Input_{train_id}", color="white", domain="DeepDriveMD_Worker"):
            input_data.dump_yaml(self.workdir / "input.yaml")

        # [PROFILE] Model Initialization (CPU)
        # Captures overhead of creating the PyTorch graph and setting up the optimizer
        with nvtx.annotate(f"Train_Init_Model_{train_id}", color="blue", domain="DeepDriveMD_Worker"):
            cvae_settings = CVAESettings.from_yaml(self.config.cvae_settings_yaml).dict()
            trainer = SymmetricConv2dVAETrainer(**cvae_settings)

        # [PROFILE] Checkpoint Loading (Disk Read + CPU-GPU Transfer)
        # This measures how long it takes to restore a previous state.
        if self.config.checkpoint_path is not None:
            with nvtx.annotate(f"Train_Load_Checkpoint_{train_id}", color="yellow", domain="DeepDriveMD_Worker"):
                checkpoint = torch.load(
                    self.config.checkpoint_path, map_location=trainer.device
                )
                trainer.model.load_state_dict(checkpoint["model_state_dict"])

        # [PROFILE] Data Loading (HEAVY I/O Bottleneck)
        # This is often the slowest part. We separate loading (Disk) from concatenating (Memory).
        with nvtx.annotate(f"Train_Data_Ingestion_{train_id}", color="red", domain="DeepDriveMD_Worker"):
            
            # 1. Load Contact Maps (Disk Intensive)
            with nvtx.annotate(f"Load_CM_Numpy_Files_{train_id}", color="red", domain="DeepDriveMD_Worker"):
                # Captures the loop of opening many small files
                cm_list = [np.load(p, allow_pickle=True) for p in input_data.contact_map_paths]
            
            # 2. Concatenate (Memory Intensive)
            with nvtx.annotate(f"Concat_CM_{train_id}", color="magenta", domain="DeepDriveMD_Worker"):
                contact_maps = np.concatenate(cm_list)

            # 3. Load RMSDs
            with nvtx.annotate(f"Load_RMSD_Files_{train_id}", color="red", domain="DeepDriveMD_Worker"):
                rmsds_list = [np.load(p) for p in input_data.rmsd_paths]
                rmsds = np.concatenate(rmsds_list)

        # [PROFILE] Training Loop (The Main GPU Event)
        # This single bar represents the entire training duration.
        # Inside this bar, you will see the jagged "Step" pattern of the GPU on the timeline.
        model_dir = self.workdir / "model"
        with nvtx.annotate(f"Train_Fit_Epochs_{train_id}", color="green", domain="DeepDriveMD_Worker"):
            trainer.fit(X=contact_maps, scalars={"rmsd": rmsds}, output_path=model_dir)

        # [PROFILE] Post-Processing Logs (Disk Write)
        with nvtx.annotate(f"Train_Save_Logs_{train_id}", color="cyan", domain="DeepDriveMD_Worker"):
            pd.DataFrame(trainer.loss_curve_).to_csv(model_dir / "loss.csv")

        # [PROFILE] Model Checkpoint Resolution (Metadata Ops)
        with nvtx.annotate(f"Train_Resolve_Checkpoint_{train_id}", color="white", domain="DeepDriveMD_Worker"):
            checkpoint_dir = model_dir / "checkpoints"
            model_weight_path = natsorted(list(checkpoint_dir.glob("*.pt")))[-1]
            
            # Adjust the path to the persistent path if using node local storage.
            model_weight_path = (
                self.persistent_dir / "model" / "checkpoints" / model_weight_path.name
            )

        output_data = CVAETrainOutput(model_weight_path=model_weight_path)
        
        # [PROFILE] Final Backup (Network I/O)
        # This measures the time to push results back to shared storage
        with nvtx.annotate(f"Train_Backup_Results_{train_id}", color="purple", domain="DeepDriveMD_Worker"):
            output_data.dump_yaml(self.workdir / "output.yaml")
            self.backup_node_local()

        return output_data
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from mdlearn.nn.models.vae.symmetric_conv2d_vae import SymmetricConv2dVAETrainer
from sklearn.neighbors import LocalOutlierFactor
import nvtx

from deepdrivemd.api import Application
from deepdrivemd.apps.cvae_inference import (
    CVAEInferenceInput,
    CVAEInferenceOutput,
    CVAEInferenceSettings,
)
from deepdrivemd.apps.cvae_train import CVAESettings


class CVAEInferenceApplication(Application):
    config: CVAEInferenceSettings

    def run(self, input_data: CVAEInferenceInput) -> CVAEInferenceOutput:
        print(">>> STARTING PROFILER CAPTURE <<<")
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        # [PROFILE] Unique ID for this inference task
        inference_id = self.workdir.name

        # [PROFILE] Metadata Logging
        with nvtx.annotate(f"Inf_Log_Input_{inference_id}", color="white", domain="DeepDriveMD_Worker"):
            input_data.dump_yaml(self.workdir / "input.yaml")

        # [PROFILE] Data Loading (Disk I/O + Memory)
        # This block reads all the numpy files. High latency here = slow storage.
        with nvtx.annotate(f"Inf_Data_Load_{inference_id}", color="red", domain="DeepDriveMD_Worker"):
            
            # Sub-task: Load Contact Maps (The heavy data)
            with nvtx.annotate(f"Inf_Load_CMs_{inference_id}", color="red", domain="DeepDriveMD_Worker"):
                contact_maps = np.concatenate(
                    [np.load(p, allow_pickle=True) for p in input_data.contact_map_paths]
                )
            
            # Sub-task: Load RMSDs & Metadata (Metadata overhead)
            with nvtx.annotate(f"Inf_Load_Metadata_{inference_id}", color="magenta", domain="DeepDriveMD_Worker"):
                _rmsds = [np.load(p) for p in input_data.rmsd_paths]
                rmsds = np.concatenate(_rmsds)
                lengths = [len(d) for d in _rmsds]  # Number of frames in each simulation
                sim_frames = np.concatenate([np.arange(i) for i in lengths])
                sim_dirs = np.concatenate(
                    [[str(p.parent)] * l for p, l in zip(input_data.rmsd_paths, lengths)]
                )
        
        assert len(rmsds) == len(sim_frames) == len(sim_dirs)

        # [PROFILE] Model Setup (CPU)
        with nvtx.annotate(f"Inf_Init_Model_{inference_id}", color="blue", domain="DeepDriveMD_Worker"):
            cvae_settings = CVAESettings.from_yaml(self.config.cvae_settings_yaml).dict()
            trainer = SymmetricConv2dVAETrainer(**cvae_settings)

            # Checkpoint Loading (Disk Read)
            checkpoint = torch.load(
                input_data.model_weight_path, map_location=trainer.device
            )
            trainer.model.load_state_dict(checkpoint["model_state_dict"])

        # [PROFILE] Inference (GPU Bound)
        # This is the most critical bar. It represents the AI actually working.
        with nvtx.annotate(f"Inf_GPU_Predict_{inference_id}", color="green", domain="DeepDriveMD_Worker"):
            embeddings, *_ = trainer.predict(
                X=contact_maps, inference_batch_size=self.config.inference_batch_size
            )
        
        # [PROFILE] Save Embeddings (Disk Write)
        with nvtx.annotate(f"Inf_Save_NPY_{inference_id}", color="yellow", domain="DeepDriveMD_Worker"):
            np.save(self.workdir / "embeddings.npy", embeddings)

        # [PROFILE] Outlier Detection (CPU Bound)
        # This runs on the CPU (Scikit-Learn). 
        # While this runs, the GPU is IDLE. If this bar is long, you are wasting GPU resources.
        with nvtx.annotate(f"Inf_LOF_Compute_{inference_id}", color="purple", domain="DeepDriveMD_Worker"):
            embeddings = np.nan_to_num(embeddings, nan=0.0)
            clf = LocalOutlierFactor(n_jobs=self.config.sklearn_num_jobs)
            clf.fit(embeddings)

        # [PROFILE] Sorting & Filtering (Pandas CPU)
        with nvtx.annotate(f"Inf_Process_Results_{inference_id}", color="orange", domain="DeepDriveMD_Worker"):
            # Get best scores and corresponding indices
            df = (
                pd.DataFrame(
                    {
                        "rmsd": rmsds,
                        "lof": clf.negative_outlier_factor_,
                        "sim_dirs": sim_dirs,
                        "sim_frames": sim_frames,
                    }
                )
                .sort_values("lof")  # First sort by lof score
                .head(self.config.num_outliers)  # Take the smallest num_outliers
                .sort_values("rmsd")  # Finally, sort by rmsd
            )

            df.to_csv(self.workdir / "outliers.csv")

        torch.cuda.profiler.stop()
        print(">>> STOPPING PROFILER CAPTURE <<<")
        return CVAEInferenceOutput(
            sim_dirs=list(map(Path, df.sim_dirs)), sim_frames=list(df.sim_frames)
        )

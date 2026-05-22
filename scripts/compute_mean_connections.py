import os
import numpy as np
from nilearn import plotting
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from rich.progress import Progress
import warnings

from src.load_data import PatientLoader
from src.process import NeuroFeatureExtractor

def main(dataset_path):
    loader = PatientLoader()
    extractor = NeuroFeatureExtractor(target_resolution=3)

    patient_dirs = [d for d in os.listdir(dataset_path) if d.startswith("sub-")]

    matrices_young = []
    matrices_adult = []
    
    with Progress() as progress:
        task = progress.add_task("[bold blue]Processing patients...", total=len(patient_dirs))
        jobs = (delayed(process_patient)(p_dir, dataset_path, loader, extractor) for p_dir in patient_dirs)
        
        for p_id, matrix in Parallel(n_jobs=6, return_as="generator")(jobs):
            progress.update(task, advance=1)
            
            if p_id is None or matrix is None: 
                continue
            
            if p_id.startswith('1'):
                matrices_young.append(matrix)
            elif p_id.startswith('2'):
                matrices_adult.append(matrix)

    plot_results(matrices_young, matrices_adult)

def process_patient(patient_dir, dataset_path, loader, extractor):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        patient_id = patient_dir.split('-')[1]
        try:
            recording = loader.load(os.path.join(dataset_path, patient_dir))
            time_series = extractor.extract_time_series(recording)
            matrix = extractor.extract_functional_connectivity(time_series)
            return patient_id, matrix
        except Exception:
            return None, None

def plot_results(matrices_young, matrices_adult):
    mean_young = np.nanmean(matrices_young, axis=0) # Using nanmean just in case
    mean_adult = np.nanmean(matrices_adult, axis=0)
    matrix_diff = mean_adult - mean_young

    fig, axes = plt.subplots(1, 3, figsize=(24, 6))

    plotting.plot_matrix(mean_young, axes=axes[0], title="Media Jóvenes (Grupo 1)",
                         vmax=1, vmin=-1, cmap='RdBu_r', colorbar=True)

    plotting.plot_matrix(mean_adult, axes=axes[1], title="Media Adultos (Grupo 2)",
                         vmax=1, vmin=-1, cmap='RdBu_r', colorbar=True)

    plotting.plot_matrix(matrix_diff, axes=axes[2], title="Diferencia (Adultos - Jóvenes)",
                         cmap='coolwarm', colorbar=True) 

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    dataset_path = "data/original"
    main(dataset_path)
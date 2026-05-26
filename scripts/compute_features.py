import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from nilearn import plotting
from rich.console import Console
from rich.progress import Progress

from src.load_data import PatientLoader
from src.process import NeuroFeatureExtractor
from src.config import CHUNK_LENGTH, STRIDE

DEFAULT_DATASET_DIR = "data/original"
DEFAULT_PROCESSED_DIR = "data/processed"
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_TARGET_RES_MM = 3
DEFAULT_N_JOBS = 6
GROUP_PLOT_NAME = "mean_functional_connectivity_groups.png"

FEATURES_ARCHIVE_SUFFIX = "_features.npz"

# Derived from participant ID (`sub-{id}`): grupo 1 (IDs starting with 1) vs grupo 2 (starting with 2).
LABEL_YOUNG = 0
LABEL_ADULT = 1
LABEL_UNKNOWN = -1


def _label_from_subject_key(subject_key: str) -> int:
    """Map `sub-{subject_key}` first digit group to a class label; ``-1`` if unrecognized."""
    if subject_key.startswith("1"):
        return LABEL_YOUNG
    if subject_key.startswith("2"):
        return LABEL_ADULT
    return LABEL_UNKNOWN


def process_single_patient(
    patient_folder: str,
    dataset_path: str,
    processed_output_dir: str,
    target_resolution_mm: float,
) -> tuple[str | None, np.ndarray | None]:
    """
    Extract features using a sliding window approach, save one ``.npz`` per window; 
    return key and global FC matrix for group plots.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loader = PatientLoader()
        extractor = NeuroFeatureExtractor(target_resolution=target_resolution_mm)
        patient_path = os.path.join(dataset_path, patient_folder)

        try:
            recording = loader.load(patient_path)
            full_time_series = extractor.extract_time_series(recording)
            global_corr_matrix = extractor.extract_functional_connectivity(full_time_series)

            subject_key = patient_folder.split("-", 1)[1]
            label = _label_from_subject_key(subject_key)
            basename = patient_folder

            n_frames = full_time_series.shape[0]
            
            window_idx = 0
            if n_frames >= CHUNK_LENGTH:
                # Iterate through the available windows per recording and calculate the window connectivity
                for start_frame in range(0, n_frames - CHUNK_LENGTH + 1, STRIDE):
                    end_frame = start_frame + CHUNK_LENGTH
                    
                    window_time_series = full_time_series[start_frame:end_frame, :]
                    window_corr_matrix = extractor.extract_functional_connectivity(window_time_series)
                    
                    archive_path = os.path.join(
                        processed_output_dir, 
                        f"{basename}_win-{window_idx:03d}{FEATURES_ARCHIVE_SUFFIX}"
                    )
                    
                    np.savez_compressed(
                        archive_path,
                        activation_time_series=window_time_series,
                        corr_matrix=window_corr_matrix,
                        label=np.int8(label),
                    )
                    
                    window_idx += 1
            else:
                # Si la serie es más corta que la ventana, la saltamos o la procesamos entera (opcional)
                raise ValueError(f"La serie temporal tiene solo {n_frames} frames, se requieren {CHUNK_LENGTH}.")

            # Devolvemos la matriz global para que la visualización final siga funcionando
            return subject_key, global_corr_matrix
            
        except Exception as e:
            # Puedes cambiar esto a un print(e) si necesitas debugear por qué falla algún sujeto
            return None, None


def plot_and_save_group_means(
    matrices_young: list[np.ndarray],
    matrices_adult: list[np.ndarray],
    output_path: str,
    console: Console,
) -> None:
    if not matrices_young:
        console.print("[yellow]No young-group subjects (IDs starting with '1'); skipping plot.")
        return
    if not matrices_adult:
        console.print("[yellow]No adult-group subjects (IDs starting with '2'); skipping plot.")
        return

    mean_young = np.nanmean(matrices_young, axis=0)
    mean_adult = np.nanmean(matrices_adult, axis=0)
    matrix_diff = mean_adult - mean_young

    fig, axes = plt.subplots(1, 3, figsize=(24, 6))

    plotting.plot_matrix(
        mean_young,
        axes=axes[0],
        title="Media Jóvenes (Grupo 1) - Global",
        vmax=1,
        vmin=-1,
        cmap="RdBu_r",
        colorbar=True,
    )
    plotting.plot_matrix(
        mean_adult,
        axes=axes[1],
        title="Media Adultos (Grupo 2) - Global",
        vmax=1,
        vmin=-1,
        cmap="RdBu_r",
        colorbar=True,
    )
    plotting.plot_matrix(
        matrix_diff,
        axes=axes[2],
        title="Diferencia (Adultos - Jóvenes)",
        cmap="coolwarm",
        colorbar=True,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"[green]Saved figure to {output_path}")


def main() -> None:
    console = Console()
    os.makedirs(DEFAULT_PROCESSED_DIR, exist_ok=True)

    patient_folders = sorted(
        d
        for d in os.listdir(DEFAULT_DATASET_DIR)
        if os.path.isdir(os.path.join(DEFAULT_DATASET_DIR, d)) and d.startswith("sub-")
    )
    console.print(
        f"[bold]Found[/bold] {len(patient_folders)} subjects under [cyan]{DEFAULT_DATASET_DIR}[/cyan]."
    )

    matrices_young: list[np.ndarray] = []
    matrices_adult: list[np.ndarray] = []

    jobs = (
        delayed(process_single_patient)(
            folder, DEFAULT_DATASET_DIR, DEFAULT_PROCESSED_DIR, DEFAULT_TARGET_RES_MM
        )
        for folder in patient_folders
    )

    with Progress() as progress:
        task = progress.add_task(
            "[bold blue]Extracting dynamic windows and global connectivity...", total=len(patient_folders)
        )
        for subject_key, matrix in Parallel(
            n_jobs=DEFAULT_N_JOBS, return_as="generator"
        )(jobs):
            progress.advance(task)

            if subject_key is None or matrix is None:
                continue

            if subject_key.startswith("1"):
                matrices_young.append(matrix)
            elif subject_key.startswith("2"):
                matrices_adult.append(matrix)

    plot_path = os.path.join(DEFAULT_OUTPUT_DIR, GROUP_PLOT_NAME)
    plot_and_save_group_means(matrices_young, matrices_adult, plot_path, console)
    console.print(f"[bold green]Data augmentation completed. Data saved in {DEFAULT_PROCESSED_DIR}[/bold green]")


if __name__ == "__main__":
    main()
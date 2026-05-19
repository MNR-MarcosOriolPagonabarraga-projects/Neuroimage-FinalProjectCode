import os
import numpy as np
from joblib import Parallel, delayed

# Import your classes
from src.load_data import PatientLoader
from src.process import NeuroFeatureExtractor

def process_single_patient(patient_path, output_dir):
    """
    Worker function. We initialize the loader and extractor INSIDE 
    the worker to avoid memory-leak and object-pickling errors across CPU cores.
    """
    try:
        loader = PatientLoader()
        extractor = NeuroFeatureExtractor(target_resolution=3, n_rois=200)
        
        patient_id = os.path.basename(os.path.normpath(patient_path))
        patient_record = loader.load(patient_path)
        
        # Extract features
        time_series = extractor.extract_time_series(patient_record)
        
        # Save directly to disk
        out_file = os.path.join(output_dir, f"{patient_id}_timeseries.npy")
        np.save(out_file, time_series)
        
        return f"Success: {patient_id}"
    
    except Exception as e:
        return f"Failed: {patient_path} | Error: {str(e)}"

if __name__ == "__main__":
    dataset_dir = "data/" # Point to your root dataset folder
    output_dir = "data/extracted_features/"
    os.makedirs(output_dir, exist_ok=True)

    # Gather all patient directories
    patient_paths = [
        os.path.join(dataset_dir, d) for d in os.listdir(dataset_dir) 
        if os.path.isdir(os.path.join(dataset_dir, d)) and d.startswith("sub-")
    ]
    
    print(f"Found {len(patient_paths)} patients. Starting parallel extraction...")

    # RUN IN PARALLEL
    # WARNING: See the RAM note below about setting n_jobs
    results = Parallel(n_jobs=4, verbose=10)(
        delayed(process_single_patient)(path, output_dir) for path in patient_paths
    )
    
    print("Batch processing complete!")
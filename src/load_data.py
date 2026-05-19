import os
import json
import nibabel as nib
import numpy as np

class Recording:
    def __init__(self, nib_img, sampling_period):
        self.sampling_period : float = sampling_period
        self.img = nib_img  # Store Nifti1Image object
    
    @property
    def data(self):
        return self.img.get_fdata()

    @property
    def affine(self):
        return self.img.affine
    
class Patient:
    name : str = None
    anat : Recording = None
    func : Recording = None


class PatientLoader:
    """
    Loads patient data from a given path. Expects a specific directory structure and file naming convention.
    Methods:
        - load(patient_path): Loads the anatomical and functional data for a patient from the specified path
    """
    def load(self, patient_path):
        patient_name = patient_path.split('/')[-1]
        patient = Patient()
        patient.name = patient_name

        for scenario in ['anat', 'func']:
            if scenario == 'func':
                json_path = os.path.join(patient_path, scenario, f"{patient_name}_task-rest_dir-AP_run-01_bold.json")
                nifti_path = os.path.join(patient_path, scenario, f"{patient_name}_task-rest_dir-AP_run-01_bold.nii.gz")
            else:
                json_path = os.path.join(patient_path, scenario, f"{patient_name}_T1w.json")
                nifti_path = os.path.join(patient_path, scenario, f"{patient_name}_T1w.nii.gz")
            data = self._load_nifti(nifti_path)
            sampling_period = self._get_metadata(json_path, 'RepetitionTime')
            patient.__setattr__(scenario, Recording(data, sampling_period))

        return patient

    def _load_nifti(self, nifti_path):
        return nib.load(nifti_path)
    
    def _get_metadata(self, json_path, key):
        with open(json_path, 'r') as f:
            js = json.load(f)
        
        return js[key]
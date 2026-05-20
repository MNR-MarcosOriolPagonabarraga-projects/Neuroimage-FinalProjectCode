import os
import json
import nibabel as nib

class Recording:
    def __init__(self, nib_img, sampling_period):
        self.sampling_period: float = sampling_period
        self.img = nib_img  
    
    @property
    def data(self):
        return self.img.get_fdata()

    @property
    def affine(self):
        return self.img.affine
    
class Patient:
    def __init__(self):
        self.name: str = None
        self.anat: Recording = None
        self.func: Recording = None

class PatientLoader:
    def __init__(self):
        pass  # Ya no necesitamos cargar ninguna plantilla MNI aquí

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
            
            raw_nifti = self._load_nifti(nifti_path)
            sampling_period = self._get_metadata(json_path, 'RepetitionTime')
            patient.__setattr__(scenario, Recording(raw_nifti, sampling_period))

        return patient

    def _load_nifti(self, nifti_path):
        return nib.load(nifti_path)
    
    def _get_metadata(self, json_path, key):
        with open(json_path, 'r') as f:
            js = json.load(f)
        return js[key]
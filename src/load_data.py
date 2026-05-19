import os
import json
import nibabel as nib
import numpy as np
from nilearn import datasets, image, masking  # Importaciones necesarias

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
        # 1. Traemos el atlas MNI152 oficial de la resolución que queremos.
        # Al poner res=3, Nilearn nos da una plantilla real de 3x3x3mm geométricamente perfecta.
        # ¡Esto ocupará muy poca memoria RAM y evitará el MemoryError!
        mni_dataset = datasets.fetch_icbm152_2009()
        self.mni_template_3mm = mni_dataset['t1']

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
            
            # 2. Resampleamos la fMRI y el T1 a la plantilla oficial de 3mm.
            # El origen geométrico no se moverá y el cerebro saldrá en su sitio.
            mni_nifti = image.resample_to_img(
                source_img=raw_nifti,
                target_img=self.mni_template_3mm,
                interpolation='linear'
            )
            
            sampling_period = self._get_metadata(json_path, 'RepetitionTime')
            patient.__setattr__(scenario, Recording(mni_nifti, sampling_period))

        return patient

    def _load_nifti(self, nifti_path):
        return nib.load(nifti_path)
    
    def _get_metadata(self, json_path, key):
        with open(json_path, 'r') as f:
            js = json.load(f)
        return js[key]
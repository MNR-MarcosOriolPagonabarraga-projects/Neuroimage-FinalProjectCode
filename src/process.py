from nilearn.maskers import NiftiLabelsMasker
from nilearn import image, datasets
import numpy as np

class NeuroFeatureExtractor:
    def __init__(self, target_resolution=3, n_rois=200, yeo_networks=7):
        self.target_affine = self._get_target_affine(target_resolution)

        print(f"Fetching Schaefer Atlas ({n_rois} ROIs)...")
        self.atlas = datasets.fetch_atlas_schaefer_2018(
            n_rois=n_rois, 
            yeo_networks=yeo_networks
        )

        self.masker = self._get_masker()

    def _get_masker(self):
        masker = NiftiLabelsMasker(
            labels_img=self.atlas.maps,
            resampling_target="data", # Forces data to resample to the atlas's space
            standardize='zscore',
            smoothing_fwhm=5,           # Applies 5mm smoothing
            detrend=True,
            high_pass=0.01,
            t_r=2.0,
            memory='nilearn_cache',     # Caches allocations
            memory_level=1
        )
        return masker
    
    def _resample(self, ni_img):
        resampled_img = image.resample_img(
            ni_img,
            target_affine=self.target_affine,
            interpolation='linear'
        )
        return resampled_img

    def extract_time_series(self, recording):
        """
        Extracts the localized average BOLD signal per ROI across time.
        Output shape: (Timepoints, Number of ROIs) -> e.g., (300, 200)
        """
        resampled_img = self._resample(recording.func.img)
        time_series = self.masker.fit_transform(resampled_img)
        return time_series

    def extract_functional_connectivity(self, time_series):
        """
        Future expansion feature: Computes a functional correlation matrix 
        from the extracted time-series data.
        Output shape: (Number of ROIs, Number of ROIs) -> e.g., (200, 200)
        """
        # Calculate the Pearson product-moment correlation coefficients
        connectivity_matrix = np.corrcoef(time_series.T)
        return connectivity_matrix


    def _get_target_affine(self, target_resolution):
        """
        Computes a single, standardized MNI grid configuration 
        scaled to the chosen voxel resolution.
        """
        # Load the standard 1mm reference template once to extract standard MNI orientation
        mni_template = datasets.load_mni152_template()
        orig_affine = mni_template.affine
        
        # Scale the directional vectors by your target resolution
        target_affine = np.copy(orig_affine)
        for i in range(3):
            scale_factor = np.linalg.norm(orig_affine[:3, i])
            target_affine[:3, i] = (orig_affine[:3, i] / scale_factor) * target_resolution
 
        return target_affine
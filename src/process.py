from nilearn.maskers import NiftiLabelsMasker
from nilearn import image, datasets
from nilearn.connectome import ConnectivityMeasure
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
            smoothing_fwhm=5,           # Applies 5mm smoothing directly
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

   
    def extract_functional_connectivity(self, recording):
        """
        Computes a functional correlation matrix from the extracted time-series data.
        Output shape: Guaranteed consistent dimensions.
        """
        
        atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm', symmetric_split=True)

        labels_masker = NiftiLabelsMasker(
            labels_img=atlas.maps, #atlas sin ajustar al cerebro
            standardize="zscore",      
            detrend=True,
            low_pass=0.1,
            high_pass=0.01,
            t_r=recording.func.sampling_period,
            memory="nilearn_cache",
            memory_level=1,
            verbose=1,
            resampling_target='labels', #de aqui para abajo son cosas para que lla matriz siempre tenga el mismo tamaño
            keep_masked_labels=True)

        region_time_series = labels_masker.fit_transform(recording.func.img)

        connectivity_measure = ConnectivityMeasure(
            kind='correlation', 
            standardize="zscore"
        )

        connectivity_matrices = connectivity_measure.fit_transform([region_time_series])
        connectivity_matrix = connectivity_matrices[0]
        
        return connectivity_matrix


    def _get_target_affine(self, target_resolution):
        """
        Computes a single, standardized MNI grid configuration 
        scaled to the chosen voxel resolution.
        """
        mni_template = datasets.load_mni152_template()
        orig_affine = mni_template.affine
        
        target_affine = np.copy(orig_affine)
        for i in range(3):
            scale_factor = np.linalg.norm(orig_affine[:3, i])
            target_affine[:3, i] = (orig_affine[:3, i] / scale_factor) * target_resolution
 
        return target_affine
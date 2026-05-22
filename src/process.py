from nilearn.maskers import NiftiLabelsMasker
from nilearn import image, datasets
from nilearn.connectome import ConnectivityMeasure
import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting

class NeuroFeatureExtractor:
    def __init__(self, target_resolution=3):
        self.target_affine = self._get_target_affine(target_resolution)
        self.atlas = datasets.fetch_atlas_harvard_oxford(
            'cort-maxprob-thr25-2mm', 
            symmetric_split=True
        )
        self.resampled_atlas = image.resample_img(
            self.atlas.maps,
            target_affine=self.target_affine,
            interpolation='nearest'
        )
        self.masker = self._get_masker()

    def _get_masker(self):
        masker = NiftiLabelsMasker(
            labels_img=self.resampled_atlas,
            standardize="zscore_sample",      
            detrend=True,
            low_pass=0.1,
            high_pass=0.01,
            t_r=2.3,
            smoothing_fwhm=5,
            memory=None,
            memory_level=0,
            verbose=0,
            resampling_target='labels', 
            keep_masked_labels=True
        )
        return masker
    
    def extract_time_series(self, recording):
        """
        Extracts the localized average BOLD signal per ROI across time.
        Output shape: (Timepoints, Number of ROIs) -> e.g., (300, 200)
        """
        self.masker.t_r = recording.func.sampling_period
        time_series = self.masker.fit_transform(recording.func.img)
        return time_series

   
    def extract_functional_connectivity(self, region_time_series):
        """
        Computes a functional correlation matrix from the extracted time-series data.
        Output shape: Guaranteed consistent dimensions .
        """
        connectivity_measure = ConnectivityMeasure(
            kind='correlation', 
            standardize="zscore_sample"
        )
        connectivity_matrices = connectivity_measure.fit_transform([region_time_series])
        
        return connectivity_matrices[0]
    
    def plot_connectivity_matrix(self):
        plot_matrix = self.connectivity_matrix.copy()
        np.fill_diagonal(plot_matrix, 0)
        
        fig, ax = plt.subplots(figsize=(26, 26))
        
        plotting.plot_matrix(
            plot_matrix,
            labels=self.atlas.labels[1:],
            cmap='RdBu_r',
            vmax=0.5,
            vmin=-0.5,
            colorbar=True,
            reorder='average',
            axes=ax   
        )
        ax.tick_params(axis='both', labelsize=6)
        
        plt.xticks(rotation=90)
        
        plt.tight_layout()
        plt.show()


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
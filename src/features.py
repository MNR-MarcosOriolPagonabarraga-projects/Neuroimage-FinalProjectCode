from nilearn.maskers import NiftiLabelsMasker

def extract_features(mni_functional_img, atlas_maps_path):
    # Initialize the masker with the atlas
    masker = NiftiLabelsMasker(
        labels_img=atlas_maps_path, 
        standardize='zscore',  # Temporal standardization
        memory="nilearn_cache", # Caches operations to save RAM
        verbose=1
    )
    
    # Extract timeseries (Number of timepoints, Number of ROIs)
    time_series = masker.fit_transform(mni_functional_img)
    
    return time_series
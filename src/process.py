from nilearn import datasets, image, masking

def preprocess(recording):
    # Smooth the image
    smoothed_img = image.smooth_img(recording.func.img, fwhm=5)
    
    # Fetch standard MNI template
    mni_template = datasets.load_mni152_template()
    
    # Resample to the MNI template space
    resliced_img = image.resample_to_img(
        smoothed_img, 
        target_img=mni_template, 
        interpolation='continuous'
    )

    # Compute and apply mask
    mask_img = masking.compute_epi_mask(resliced_img)
    final_data = masking.apply_mask(resliced_img, mask_img)

    return final_data, resliced_img
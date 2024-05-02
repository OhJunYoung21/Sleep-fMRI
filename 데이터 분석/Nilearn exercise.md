### Simply visualizing data(NifTi format)

~~~python3
import nilearn
from nilearn import plotting
file_path = "/Users/ojun-yong/Desktop/bids_ex/fmriprep/sub-01/ses-1/anat/sub-01_ses-1_acq-Sagittal3DT1GRE_rec-BRAINMRINONCONTRASTDIFFUSION_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz"
plotting.plot_img(file_path)
~~~

### Smoothing preprocessed data manually


~~~python3
import os
import glob
from pathlib import Path

directory_path = "/Users/ojun-yong/Desktop/bids_ex/fmriprep/sub-01/ses-1/anat"
search_string = "Sagittal3DT1GRE"
end_string = ".nii.gz"

all_files = glob.glob(os.path.join(directory_path,"*"))

target_files = [file for file in all_files if file.endswith(end_string)]

anat_all = image.smooth_img(target_files,fwhm=5)
~~~

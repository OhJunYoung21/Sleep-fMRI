import numpy as np
from nilearn import datasets
from nilearn.maskers import NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn import plotting
from nilearn import datasets
import os
import nibabel as nib
from nilearn import masking
from nilearn import image
from nilearn import input_data
from nipype.interfaces import afni
from scipy import stats

# Download the Shen atlas
atlas_path = '/Users/oj/Desktop/Yoo_Lab/atlas/shen_2mm_268_parcellation.nii'

file_path = '/Users/oj/Desktop/Yoo_Lab/post_fMRI/confounds_regressed_RBD/sub-01_confounds_regressed.nii.gz'


def FC_extraction(file_path, atlas_path):
    shen_atlas = input_data.NiftiLabelsMasker(labels_img=atlas_path, standardize=True)

    data = image.load_img(file_path)

    time_series = shen_atlas.fit_transform(data)

    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([time_series])[0]

    return correlation_matrix


## calculate_3dReHo는 AFNI의 3dReHo를 사용해서 input으로는 4D image를 받고 output으로 3d image를 반환한다.
def calculate_3dReHo(file_path, output_name: str):
    reho = afni.ReHo()

    reho.inputs.in_file = file_path
    reho.inputs.out_file = f'/Users/oj/Desktop/Yoo_Lab/post_fMRI/confounds_regressed_RBD/reho/reho_{output_name}.nii.gz'

    result = reho.run()

    result_path = f'/Users/oj/Desktop/Yoo_Lab/post_fMRI/confounds_regressed_RBD/reho/reho_{output_name}.nii.gz'

    return result_path


## region_reho_average는 mask가 나눈 region안의 voxel 값들의 평균을 계산한다.
def region_reho_average(reho_file, atlas_path):
    shen_atlas = input_data.NiftiLabelsMasker(labels_img=atlas_path, standardize=True, strategy='mean')

    reho_img = image.load_img(reho_file)

    masked_data = shen_atlas.fit_transform([reho_img])

    return masked_data


def calculate_Bandpass(file_path, output_name: str):
    Bandpass = afni.Bandpass()
    Bandpass.inputs.in_file = file_path
    Bandpass.inputs.highpass = 0.01
    Bandpass.inputs.lowpass = 0.1

    # 파일을 저장하는 경로는 out_file로 지정하며, out_file코드를 실행한다는 것은 파일을 저장하겠다는 의미이다.

    Bandpass.inputs.out_file = f'/Users/oj/Desktop/Yoo_Lab/post_fMRI/confounds_regressed_RBD/alff/alff_{output_name}.nii.gz'

    Bandpass.run()

    alff_path = f'/Users/oj/Desktop/Yoo_Lab/post_fMRI/confounds_regressed_RBD/alff/alff_{output_name}.nii.gz'

    alff_data = image.load_img(alff_path).get_fdata()

    x, y, z, t = alff_data.shape

    alff_img = np.empty((x, y, z))

    for x in range(x):
        for y in range(y):
            for z in range(z):
                time_series = alff_data[x, y, z, :]
                if np.mean(time_series) != 0:
                    alff_img[x, y, z] = np.sum(time_series ** 2) / np.mean(time_series)
                else:
                    alff_img[x, y, z] = 0.0
    '''
    alff_img = stats.zscore(alff_img, axis=0)
    '''

    alff_nifti = nib.Nifti1Image(alff_img, None)

    nib.save(alff_nifti,
             f'/Users/oj/Desktop/Yoo_Lab/post_fMRI/confounds_regressed_RBD/alff/alff_transformed_{output_name}.nii.gz')

    result_path = f'/Users/oj/Desktop/Yoo_Lab/post_fMRI/confounds_regressed_RBD/alff/alff_transformed_{output_name}.nii.gz'

    return result_path


def region_alff_average(alff_path, atlas_path):
    shen_atlas = input_data.NiftiLabelsMasker(labels_img=atlas_path, standardize=True, strategy='mean',
                                              resampling_target="labels")

    alff_img = image.load_img(alff_path)

    masked_data = shen_atlas.fit_transform([alff_img])

    return masked_data


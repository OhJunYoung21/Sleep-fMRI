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
from scipy.stats import kendalltau
from nipype.interfaces import afni

# Download the Shen atlas
atlas_path = '/Users/oj/Downloads/shen_2mm_268_parcellation.nii'

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

    img = image.load_img(result_path)

    return img


## region_reho_average는 mask가 나눈 region안의 voxel 값들의 평균을 계산한다.
def region_reho_average(reho_file, atlas_path):
    shen_atlas = input_data.NiftiLabelsMasker(labels_img=atlas_path, standardize=True, strategy='mean')

    reho_img = image.load_img(reho_file)

    masked_data = shen_atlas.fit_transform([reho_img])

    return masked_data


def Bandpass_transform(file_path, output_name: str):
    bandpass = afni.Bandpass()
    bandpass.inputs.in_file = file_path
    bandpass.inputs.highpass = 0.01
    bandpass.inputs.lowpass = 0.1
    bandpass.inputs.out_file = f'/Users/oj/Desktop/Yoo_Lab/post_fMRI/confounds_regressed_RBD/alff/alff_{output_name}.nii.gz'

    bandpass.run()

    result_path = f'/Users/oj/Desktop/Yoo_Lab/post_fMRI/confounds_regressed_RBD/alff/alff_{output_name}.nii.gz'
    img = image.load_img(result_path)

    data_4d = img.get_fdata()

    x, y, z, t = img.shape

    bandpass_result = np.empty((x, y, z))

    for x in range(x):
        for y in range(y):
            for z in range(z):
                time_series = data_4d[x, y, z, :]
                if np.mean(time_series) != 0:
                    bandpass_result[x, y, z] = np.sum(time_series) / np.mean(time_series)
                else:
                    bandpass_result[x, y, z] = 0

    ### 각 voxel은 filtering된 시계열 데이터를 가지고 있고, 해당 4d 데이터를 .nii.gz 형태로 바꿔서 저장한다.

    # affine 변환을 수행하지 않는다.
    nifti_img = nib.Nifti1Image(bandpass_result, None)

    nib.save(nifti_img,
             f'/Users/oj/Desktop/Yoo_Lab/post_fMRI/confounds_regressed_RBD/alff/alff_series_{output_name}.nii.gz')

    print(bandpass_result)

    return


def region_reho_average(reho_file, atlas_path):
    shen_atlas = input_data.NiftiLabelsMasker(labels_img=atlas_path, standardize=True, strategy='mean')

    reho_img = image.load_img(reho_file)

    masked_data = shen_atlas.fit_transform([reho_img])

    return masked_data


Bandpass_transform(file_path, "start_4")

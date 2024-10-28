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

file_path = '/Users/oj/Desktop/Yoo_Lab/post_fMRI/confounds_regressed_RBD/sub-01_confounds_regressed.nii.gz'

craddock = datasets.fetch_atlas_craddock_2012()
atlas_filename = craddock.maps

data = image.load_img(file_path)

craddock_atlas = input_data.NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)

sample = craddock_atlas.fit_transform(data)

print(sample.shape)


def FC_extraction(file_path):
    craddock_atlas = input_data.NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)

    data = image.load_img(file_path)

    time_series = craddock_atlas.fit_transform(data)

    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([time_series])[0]

    return correlation_matrix


## calculate_3dReHo는 AFNI의 3dReHo를 사용해서 input으로는 4D image를 받고 output으로 3d image를 반환한다.


'''
건강군과 질병군마다 분류기준을 추출한다. 경로를 헷갈리지 않게 하기 위해서 feature 추출하는 함수를 2개씩 작성하였다.
'''

'''
ReHo를 계산한다.
'''


## region_reho_average는 mask가 나눈 region안의 voxel 값들의 평균을 계산한다.
def region_reho_average(reho_file, atlas_path):
    craddock_atlas = input_data.NiftiLabelsMasker(labels_img=atlas_filename, standardize=True, strategy='mean')

    reho_img = image.load_img(reho_file)

    masked_data = craddock_atlas.fit_transform([reho_img])

    return masked_data


def calculate_Bandpass(file_path, output_name: str, root_dir: str):
    Bandpass = afni.Bandpass()
    Bandpass.inputs.in_file = file_path
    Bandpass.inputs.highpass = 0.01
    Bandpass.inputs.lowpass = 0.1

    # 파일을 저장하는 경로는 out_file로 지정하며, out_file코드를 실행한다는 것은 파일을 저장하겠다는 의미이다.

    ## out_file = alff_path, 위 둘의 경로는 동일하다.ALFF의 첫번째 결과물은 alff_path에 output_name을 추가해서 저장해준다.

    Bandpass.inputs.out_file = os.path.join(root_dir, f'alff_{output_name}.nii.gz')

    Bandpass.run()

    alff_data = image.load_img(os.path.join(root_dir, f'alff_{output_name}.nii.gz')).get_fdata()

    x, y, z, t = alff_data.shape

    alff_img = np.empty((x, y, z))

    for x in range(x):
        for y in range(y):
            for z in range(z):
                time_series = alff_data[x, y, z, :]
                if np.mean(time_series) != 0:
                    alff_img[x, y, z] = np.sum(time_series ** 2)
                else:
                    alff_img[x, y, z] = 0.0
    '''
    alff_img = stats.zscore(alff_img, axis=0)
    '''

    alff_nifti = nib.Nifti1Image(alff_img, None)

    end_path = os.path.join(root_dir, f'alff_transformed_{output_name}.nii.gz')

    nib.save(alff_nifti,
             end_path)

    return


def region_alff_average(alff_path, atlas_path):
    craddock_atlas = input_data.NiftiLabelsMasker(labels_img=atlas_filename, standardize=True, strategy='mean',
                                                  resampling_target="labels")

    alff_img = image.load_img(alff_path)

    masked_data = craddock_atlas.fit_transform([alff_img])

    return masked_data

from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
import os
from nilearn import image
from nilearn import plotting


def FC_extraction(file_path):
    basc_atlas = datasets.fetch_atlas_basc_multiscale_2015()
    basc_atlas_name = basc_atlas["scale036"]

    data = image.load_img(file_path)

    masker = NiftiLabelsMasker(labels_img=basc_atlas_name, standardize=True)
    time_series = masker.fit_transform(data)

    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([time_series])[0]

    return correlation_matrix


def ALFF_extraction(file_path):
    return


def Reho_extraction(file_path):
    return

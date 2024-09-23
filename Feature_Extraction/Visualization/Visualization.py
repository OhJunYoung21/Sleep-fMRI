from Feature_Extraction.Shen_features.Classification_feature import FC_extraction
from Feature_Extraction.Shen_features.Classification_feature import atlas_path
from Feature_Extraction.Shen_features.Classification_feature import file_path
from nilearn import plotting
from nilearn import input_data
from Feature_Extraction.Shen_features.Classification_feature import calculate_3dReHo, region_reho_average
import nilearn
from nilearn import image

atlas = image.load_img(atlas_path)


def FC_show():
    Connectivity = FC_extraction(file_path, atlas_path)

    plotting.plot_matrix(Connectivity)
    plotting.show()

    return


def ReHo_show():
    path = calculate_3dReHo(file_path, "visualize")
    result = region_reho_average(path, atlas_path)

    plotting.plot_matrix(result)
    plotting.show()

    print(result.shape)
    return

ReHo_show()
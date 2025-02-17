import glob
import os
from Comparison_features.rsfmri import static_measures
import re
from Static_Feature_Extraction.Shen_features.DataStructure import input_fc, make_reho_shen, make_alff_shen, \
    make_falff_shen
import pandas as pd

file_path = '/Users/oj/Desktop/Yoo_Lab/Yoo_data/RBD_PET_classification/RBD_PET_positive_regressed'
sub_dir = "/Users/oj/Desktop/Yoo_Lab/Yoo_data/RBD_PET_classification/positive_MNI152NLin2009cAsym_mask"



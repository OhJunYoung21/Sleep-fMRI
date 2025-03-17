from nilearn import datasets, plotting, image
import numpy as np
import os
import nibabel as nib
import pandas as pd
from nilearn.datasets import load_mni152_template
from PET_classification.result_analysis import count_occurrences, find_region

### bring shen_atlas file manually

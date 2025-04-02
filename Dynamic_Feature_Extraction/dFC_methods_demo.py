from pydfc import data_loader
import numpy as np
import pandas as pd
from pydfc.dfc_methods import SLIDING_WINDOW

# upload single subject's data

BOLD = data_loader.nifti2timeseries(
    nifti_file="sample_data/sub-0001_task-restingstate_acq-mb3_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
    n_rois=100,
    Fs=1 / 3,
    subj_id="sub-0001",
    confound_strategy="no_motion",  # no_motion, no_motion_no_gsr, or none
    standardize=False,
    TS_name=None,
    session=None,
)

params_methods = {
    "W": 9,
    "n_overlap": 0.6,
    "sw_methods": "pear_corr",
    "tapered_window": True,
    "normalization": True,
    "num_select_nodes": None
}

measure = SLIDING_WINDOW(**params_methods)
dFC = measure.estimate_dFC(time_series=BOLD)

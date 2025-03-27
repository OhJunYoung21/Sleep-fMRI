import numpy as np
import pandas as pd
import nilearn
from nilearn import image
import nibabel as nib
import netplotbrain
from templateflow import api as tf

template = tf.get('MNI152NLin2009Asym', suffix="T1w")

netplotbrain.plot(template=template)

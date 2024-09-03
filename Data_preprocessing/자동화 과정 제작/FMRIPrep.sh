#!/bin/bash

input_folder="$HOME/Desktop/pre_BIDS/pre_BIDS_RBD"
output_folder="$HOME/Desktop/pre_BIDS/pre_prep_RBD"


fmriprep-docker $input_folder $output_folder \
  participant \
    --skip-bids-validation \
    --fs-license-file $HOME/Downloads/license.txt \
    --mem-mb 8000 \
    --fs-no-reconall \
    --stop-on-first-crash \
    --md-only-boilerplate \
    --use-syn-sdc \
    --dummy-scans 3 \
    --output-spaces MNI152NLin2009cAsym \
    --nthreads 8 \
    --cifti-outputs \
    
  


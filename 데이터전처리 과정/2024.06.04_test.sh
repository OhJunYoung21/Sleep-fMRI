#User inputs:
bids_root_dir=$HOME/Desktop/pre_fMRI_HC
subj=02
container=docker #docker or singularity



fmriprep-docker $bids_root_dir $bids_root_dir/fmriprep \
  participant \
    --participant-label $subj \
    --skip-bids-validation \
    --fs-license-file $HOME/Downloads/license.txt \
    --mem-mb 8000 \
    --fs-no-reconall \
    --output-spaces MNI152NLin2009cAsym \
    --nthreads 16 \
    -w $HOME 

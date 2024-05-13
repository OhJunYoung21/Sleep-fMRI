#User inputs:
bids_root_dir=$HOME/Desktop/bids_ex
subj=02
nthreads=16
container=docker #docker or singularity



fmriprep-docker $bids_root_dir $bids_root_dir/fmriprep \
  participant \
    --participant-label $subj \
    --skip-bids-validation \
    --md-only-boilerplate \
    --fs-license-file $HOME/Downloads/license.txt \
    --fs-no-reconall \
    --output-spaces MNI152NLin2009cAsym \
    --nthreads $nthreads \
    --stop-on-first-crash \
    --mem_mb 8000 \
    -w $HOME


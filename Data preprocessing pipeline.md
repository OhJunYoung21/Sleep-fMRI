## fMRIprep-docker

필자는 받은 데이터를 bidsify해준 다음, fmriprep을 사용해서 데이터 전처리 과정을 수행하였다.

각종 연구에서 데이터를 전처리하는데 사용되는 절차들이 있는데, 사용자가 수행할 단계는 아래와 같다.(교수님과 상의후 바뀔 수 있음.)

|전처리 목적| 명렁 코드 |코드 설명|   
|---|---|---|
| 초반 이미지 제거  | --dummy-scan N | 초반 이미지는 여러가지 요인으로 오염된 이미지일 수 있기 때문에 제거해준다.|
| 머리 움직임이 과한 volume 제거 | --fd-spike-threshold | framewise-displacement가 특정 수치이상인 volume을 제거한다. |  
| Slice Timing Correction |       | .json파일에 slice timing correcion 관련정보가 있으면 수행한다. |
|   despiking   |  --dvars-spike-threshold    |  DVARS는 시간의 흐름에 따른 변화율을 말하는 것으로 변화율이 갑작스럽게 변하는 경우의 volume은 제거한다.     |
|  Compcor    | FSL's melodiac-ica    |  BOLD signal에 영향을 미치는 여러 잡음 요인들을 제거한다.     |
|  공간정규화    | --output-spaces    |  그룹사이에서 개인간의 비교를 위해서 뇌의 크기를 통일시킨다.(하나의 tamplate으로 통일시킨다.)     |


### 실제 사용할 코드(프로토 타입)

~~~unix
#User inputs:
bids_root_dir=$HOME/Desktop/bids_output
subj=02
nthreads=2
container=docker #docker or singularity


fmriprep-docker $bids_root_dir $bids_root_dir/fmriprep \
  participant \
    --participant-label $subj \
    --skip-bids-validation \
    --md-only-boilerplate \
    --fs-license-file $HOME/Downloads/license.txt \
    --dummy-scans 5 \
    --bold2t1w-dof 9 \
    --fd-spike-threshold 0.3 \
    --dvars-spike-threshold \
    --fs-no-reconall \
    --output-spaces MNI152NLin2009cAsym:res-2 \
    --nthreads $nthreads \
    --stop-on-first-crash \
    --mem_mb 16000 \
    -w $HOME
~~~

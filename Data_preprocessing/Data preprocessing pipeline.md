## fMRIprep-docker

필자는 받은 데이터를 bidsify해준 다음, fmriprep을 사용해서 데이터 전처리 과정을 수행하였다.

각종 연구에서 데이터를 전처리하는데 사용되는 절차들이 있는데, 사용자가 수행할 단계는 아래와 같다.(교수님과 상의후 바뀔 수 있음.)

|전처리 목적| 명렁 코드 |코드 설명|   
|---|---|---|
| 초반 이미지 제거  | --dummy-scan N | 초반 이미지는 여러가지 요인으로 오염된 이미지일 수 있기 때문에 제거해준다.AFNI's 3d shift는 초반이미지 N개를 제거한다. |
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


fmriprep-docker $input_folder $output_folder \
  participant \
    --skip-bids-validation \
    --fs-license-file $HOME/Downloads/license.txt \
    --mem-mb 8000 \
    --fs-no-reconall \
    --stop-on-first-crash \
    --md-only-boilerplate \
    --use-syn-sdc warn \
    --dummy-scans 3 \
    --output-spaces MNI152NLin2009cAsym \
    --nthreads 8 \
~~~

| 코드  | 사용한 이유 |
|---|---|
| --skip-bids-validation  |  https://bids-standard.github.io/bids-validator/ 통해서 해당 데이터셋이 BIDS fortmat임을 확인한다. bidscoin으로 이미 bidsify가 된 것을 알기 때문에 쓸모없는 연산시간을 줄이기 위해 넣어주었다. |
| --dummy-scans 5 | 초반 5개의 이미지를 전처리 대상에 포함시키지 않는다.미리 이미지 개수를 정해놓지 않으면 각 subject별로 다른 개수의 이미지를 폐기한다.다른연구에서는 3,5,7과 같은 개수를 설정해놓는다. |
| --bold2t1w-dof 9 | coregistration과정에서 사용할 매개변수의 개수이다. defaul값은 6개이며, 개수가 늘어날수록 계산량이 늘어난다는 단점이 있다. |
| --fd-spike-threshold 0.3 | defaulat값은 0.5이지만 조금 더 기준을 낮게 설정하면 조금의 두뇌움직임만 감지되도 해당 volume은 전처리대상에 제외한다.|
| --output-spaces MNI152NLin2009cAsym:res-2  | 공간정규화에 어떤 틀을 쓸지를 정하는 코드이다.  |
| --mem_mb 16000  | 전처리 분석에 사용할 메모리 용량이다. |



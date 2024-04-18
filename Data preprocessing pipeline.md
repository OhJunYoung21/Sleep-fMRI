## fMRIprep-docker

필자는 받은 데이터를 bidsify해준 다음, fmriprep을 사용해서 데이터 전처리 과정을 수행하였다.

각종 연구에서 데이터를 전처리하는데 사용되는 절차들이 있는데, 사용자가 수행할 단계를 아래와 같다.(교수님과 상의후 바뀔 수 있음.)

|전처리 목적| 명렁 코드 |코드 설명|   
|---|---|---|
| 초반 이미지 제거  | --dummy-scan N | 초반 이미지는 여러가지 요인으로 오염된 이미지일 수 있기 때문에 제거해준다.|
| 머리 움직임 교정 | --fd-spike-threshold | framewise-displacement가 특정 수치이상인 volume을 제거한다. |  
| Slice Timing Correction |       | .json파일에 slice timing correcion 관련정보가 있으면 수행한다. |

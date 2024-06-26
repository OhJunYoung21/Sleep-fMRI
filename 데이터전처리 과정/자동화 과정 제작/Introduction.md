## 전처리 작업 자동화하기

### 1. BIDScoin에 맞는 데이터구조 만들어주기

[for BIDScoin structure](https://bidscoin.readthedocs.io/en/stable/preparation.html)

위의 링크를 보면 BIDScoin에 맞는 데이터구조가 나온다. 데이터구조에 대해서 간단하게 설명하자면, BIDScoin안에 데이터를 넣을떄, 파일을 마구잡이로 집어넣는 것이 아닌, 정리를 해서 넣어야 한다는 의미이다.

예를 들어 한 피실험자의 데이터가 [a_BOLD.nii, a_t1w.nii,a_BOLD.json,a_t1w.json] 이렇게 있다고 하자. 이때, 저 파일을 그대로 BIDScoin에 넣으면 에러가 발생한다. 필자는 이를 BIDScoin에 맞는 형식이 아니기때문이라고 이해하였다.

마치 맞지 않는 옷을 입힌 것과 같은 원리이다. 발이 275인사람한테 230을 입히면 안되는 것처럼.

그럼, 이제 구조에 맞게 바꿔줘야 하는데, 물론 파일의 개수가 몇개 되지 않으면 노가다를 좀 뛰면 된다. 하지만 파일의 개수가 수천개라면....노가다 뛰다가 팔에 쥐가 먼저 날지도 모른다.

아래의 그림과 같은 작업을 반복해서 거쳐야 하는데, 이를 .sh파일로 자동화 시켜놓으면 편할 것 같아 .sh파일을 작성해보게 되었다.

![스크린샷 2024-06-20 오후 12 45 41(2)](https://github.com/OhJunYoung21/Sleep-fMRI/assets/81908471/e181c797-c393-4f79-9bf7-7c3c45e68fc3)

위의 사진에 대해서 좀더 자세히 설명을 하자면, PAR/REC파일은 fMRI파일과 T1w 파일이 같이 제공되는 경우가 많다. 피실험자 1명에 관한 파일이 4개가 존재하는데, 이를 sub-01,sub-02같은 피실험자의 폴더에 해당 파일을 넣어주는 작업이다.

sub-01 : [a_BOLD.nii,a_BOLD.json,a_t1w.nii,a_t1w.json] , 이런식으로 이해하면 될 것이다. 이해가 되지 않는다면 언제든지 문의 주길 바란다.😃

#### 자동화 툴 만드는 단계:

* fMRI,T1w폴더 안의 파일의 개수를 구하고, 해당 개수의 절반만큼의 sub-0* 폴더를 만든다.sub-0*폴더는 pre_BIDS라는 폴더안에 만든다.
* fMRI,T1w폴더에서 가장 첫번째 파일부터 2개씩 pre_BIDS안의 폴더에 순서대로 넣는다.(ex. 맨 앞의 2개는 sub-01,그 다음 2개는 sub-02...)
* 

### BIDScoin 실행하기

[bidscoin 실행 파일](https://github.com/OhJunYoung21/Sleep-fMRI/tree/main/BIDS_Coin)

위 링크안에 bidsmapper.sh와 bidscoiner.sh를 실행하면 BIDS_format을 만들 수 있다.이때, 제대로 bids_format이 만들어졌는지를 확인하기 위해서는 bids validator를 사용하면 된다.

⬇️ 링크는 아래 참조.⬇️

[Bids validator](https://bids-standard.github.io/bids-validator/)

### fmriprep 실행하기

자동화 툴 제작 폴더 안에 들어있는 fmriprep.sh를 실행하면 된다.




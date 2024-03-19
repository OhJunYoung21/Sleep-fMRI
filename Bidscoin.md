## Bidscoin이란?

Bidscoin이란, DICOM,PAR(REC)과 같은 raw data 양식의 파일을 BIDS형식으로 바꿔주는 툴이다. GUI로 작동되며, 이떄 우리는 PyQt를 사용한다.

### PyQt란?

PyQt란. C++로 작성된 패키지를 파이썬 환경에서 작동할 수 있게 도와주는 도구라고 생각하면 된다. 마치 크로스 플랫폼처럼.

--

### BIDScoin Workflow

~~~python3
bidsmapper -> bidseditor -> bidscoiner
~~~

* bidsmapper의 역할 : source data를 BIDS format으로 만들때에는 틀을 만들어 놓고 그 틀 안에 원래의 데이터를 넣는데, 그 틀을 만들어 주는 거이 바로 bidsmapper이다. mapper를 생각해보면 map=지도 라는 것을 생각하면 soucedata가 따라가야 하는 지도를 만들어준다고 생각하면 될 것이다.

* bidseditor의 역할 : 여기서는 BIDS data가 어떤 식으로 저장될지 등을 검사하고 편집하는 단계이다. 필자는 위 단계가 핵심이라고 보는데, 여기서 각종 오류등을 사전에 방지할 수 있기 때문이다. bidsditor의 결과로 bidsmap.yaml이 생성되고, 이는 bidscoiner가 사용한다. bidscoiner는 bidsmap.yaml이 제공하는 정보를 토대로 BIDS format을 완성한다.

*  bidscoiner의 역할 : bidsmapper가 생성한 틀(지도)를 사용해서 source_data를 BIDS data format으로 바꿔주는 역할이다.😀

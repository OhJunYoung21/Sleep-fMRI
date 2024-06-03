## 오류 해결 기록

### dataset_description.json is missing

해당 오류는 dataset_decription.json파일이 어디있는지를 찾지 못해서 발생한 오류이며, 나의 경우에는 input_bids의 경로에 상대경로를 지정해서 생긴문제였다.

상대경로는 가급적 지양하는 것이 좋을듯 하며, ~/ 대신 $HOME을 사용해서 경로를 지정해주었더니 해결되었다.

### .TaskName should be a string

func파일의 .json파일을 보면 "TaskName": null이라는 부분이있었는데, 해당 부분의 null을 string으로 바꿔줘야 한다. 필자의 경우에는 "rest"로 바꿔주었더니 오류가 해결되었다.

위 오류는 bids-validator과정에서 발생하였으며, taskname에 string이 들어가있지 않으면, bids format이 아니라고 간주한다.

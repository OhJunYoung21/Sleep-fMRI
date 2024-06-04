## 오류 해결 기록

### dataset_description.json is missing

해당 오류는 dataset_decription.json파일이 어디있는지를 찾지 못해서 발생한 오류이며, 나의 경우에는 input_bids의 경로에 상대경로를 지정해서 생긴문제였다.

상대경로는 가급적 지양하는 것이 좋을듯 하며, ~/ 대신 $HOME을 사용해서 경로를 지정해주었더니 해결되었다.

---

### .TaskName should be a string

func파일의 .json파일을 보면 "TaskName": null이라는 부분이있었는데, 해당 부분의 null을 string으로 바꿔줘야 한다. 필자의 경우에는 "rest"로 바꿔주었더니 오류가 해결되었다.

위 오류는 bids-validator과정에서 발생하였으며, taskname에 string이 들어가있지 않으면, bids format이 아니라고 간주한다.

---

### zsh:command not found: fmriprep-docker

해당 오류는 fmriprep-docker의 실행파일을 찾지 못해서 발생하는 오류이다. 일시적인 해결방안으로는 경로를 매번 터미널을 킬 떄마다 추가해주면 된다.

~~~linux
export PATH="/Users/oj/library/python/3.9/bin:$PATH"
~~~

하지만 이렇게 하면, 터미널을 끄고 킬때마다 매번 다시 입력해줘야 하는 번거로움이 있다.😂

그렇기 때문에 shell 파일에 영구적으로 경로를 지정해놓는 방법을 추천한다.

1.echo 문을 통해 사용중인 쉘을 알아본다.

~~~linux
echo $SHELL
~~~

2.bin/zsh이면 ~/.zshrc 파일을 수정한다.

2-1. zsh파일을 열어준다.

~~~linux
vim ~/.zshrc
~~~

2-2. export PATH="/Users/oj/library/python/3.9/bin:$PATH" 를 zsh파일에 입력한다. i를 누르고 입력해주면 된다.

2-3. :wq를 입력한 다음, 아래의 명령어를 입력하면 된다.

~~~linux
source ~/.zshrc
~~~

위 과정을 마치면 터미널이 꺼졌다 켜져도 무리없이 fmriprep-docker 명령어를 사용할 수 있다. command not found 에러는 종종발생하니 자주 사용하는 명령어는 쉘 파일에 경로를 영구적으로 설정해놓는 편이 좋다.

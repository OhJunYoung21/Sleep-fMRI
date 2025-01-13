## 이슈사항

### PyQt6 plugin cocoa not found

~~~unix
bidscoin -t
~~~

위 명령어를 실행하면 cocoa plugin을 찾을 수 없다는 에러가 항상 등장했다. 위 문제로 상당히 많은 시간을 갑질했으나...결국 경로를 재설정해줌으로써 문제를 해결했다.

~~~unix
export QT_PLUGIN_PATH=$HOME/opt/anaconda3/lib/python3.11/site-packages/PyQt6/Qt6/plugins
~~~

QT_PLUGIN_PATH 경로에 적절한 플러그인의 위치를 설정하지 않는다면 계속 아래와 같은 에러메세지가 출력된다.

~~~unix
qt.qpa.plugin: Could not find the Qt platform plugin "cocoa" in ""
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

zsh: abort      python main.py
~~~

위 메세지를 필자는 cocoa plugin을 찾아오지 못하는 것으로 해석하였고, cpu가 찾아올 수 있도록 길을 알려준 것이다.( 위 과정에 소요된 시간 자그마치 3일...😫)

### 경로설정 문제 : zsh:command not found : bidscoin

~~~unix
export PATH="/Users/oj/Library/Python/3.9/bin":$PATH    
~~~
zsh:command not found : bidscoin 이라는 오류메세지가 뜨면 지정된 경로에서 명령어를 못찾았다는 의미이니, 경로에 명령어를 설정해주면 된다.


### bidsmap.yaml 파일 문제

~~~linux
bash bidscoiner.sh 
~~~

위 명령어를 실행할때, No bidsmap file is found라는 에러가 종종발생한다. 해당 파일은 code > bidscoin안에 들어있어야 하는데, bidsmapper결과 생성된 파일을 수동으로 넣어주면 문제가 해결된다.














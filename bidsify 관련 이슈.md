## bidsfity 이슈 해결사항

yaml.load() -> yaml.full_load()

에러 메세지 : load() missing 1 required positional argument: 'Loader

bidsify는 yaml파일을 필요로 한다. 그래서 bidsify를 실핼할때 .yml파일의 경로를 넣어주는데, 이때 단순히 load를 쓰면 위의 오류가 발생하였다. load 대신 full_load()를 사용하였더니 해당 오류가 발생하지 않았다!!!

## feature_selection process

### feature_selection based on prior knonwlegde

데이터들을 보고 여러가지 통계적 기법을 써서(two-sample t test, Lasso regression) 서로 차이가 나타나는 region(node)를 찾는것도 물론 도움이 되지만, 이전에 연구에서 정상군과 질병군 사이에서 확연한 차이가 나타난다고 알려진 부분만 집중적으로 봐서 region을

추출하는 것도 하나의 방법이다.

feature-selection을 하는 이유는 과적합을 방지하기위함이기 때문에, 위과정에서는 필수적이라고 생각하였다.

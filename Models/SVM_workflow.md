## Workflow of SVM modeling

### 훈련데이터와 시험데이터의 분류

1차적으로, train data와 test data를 분류한다.

~~~python3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
~~~

### 훈련데이터내부에서 또다시 훈련데이터와 시험데이터를 나눠준다.

X_train에는 train_test_split이 내놓은 train data가 들어있다.

나의 목적은 단순히 ALFF,REHO,fALFF로 예측모델링을 하는 것보다는 어떤 노드(region)이 예측에 많은 기여를 하는지 알아보고, 

추후에는 RBD와 HC의 각 Feature별 그룹의 평균을 계산하고 시각화 한다음 내가 추출한 노드(region)에 해당하는 부분이 실제로 차이를 보이는지 보는것이다.



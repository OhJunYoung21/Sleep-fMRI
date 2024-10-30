## 내가 겪은 오류들

### Resampling Error

~~~python3
UserWarning: After resampling the label image to the data image, the following labels were removed:
~~~

해결 방법은 아래와 같습니다.

### resampling_target{“data”, “labels”, None},

default=”data”

Gives which image gives the final shape/size. 

For example, if resampling_target is "data", the atlas is resampled to the shape of the data if needed.

If it is "labels" then mask_img and images provided to fit() are resampled to the shape and affine of maps_img.

"None" means no resampling: if shapes and affines do not match, a ValueError is raised.

참조 링크 :(https://nilearn.github.io/stable/modules/generated/nilearn.maskers.NiftiLabelsMasker.html)

---


### CSV파일에서 데이터를 불러들일때는 문자열로 데이터를 읽어들인다.

Data Features(ALFF,Reho,Functional Connectivity)를 추출해서 csv 파일로 저장하였다.(매번 파이썬 코드를 실행해서 데이터를 얻어오는 과정이 귀찮았기 때문에)

csv파일에서 데이터를 읽어오려면 pd.read_csv()를 해야 하는데, 이때 데이터는 문자열 형태로 들어오게 된다.

문자열 형태로 들어온 데이터는 모델 트레이닝에 쓰일수 없기 때문에 numpy.ndarray형태또는 리스트 형태로 바꿔줘야 한다.

~~~python3
import ast
~~~

---

### 잘못된 데이터로 인해 잘못된 결과를 초래

fALFF와 ReHo를 기준으로 했을때, 분류 성능이 1.0이 반복적으로 나오는 것을 보고 혹시 코드가 잘못되지는 않았을까 하는 생각에 코드를 살펴보다가 이상한점을 발견하였다.

~~~python3
for j in range(len_rbd):
    schaefer_data.loc[j] = [FC_RBD[j], ALFF_RBD[j], ReHo_RBD[j], fALFF_RBD[j], 1]

for k in range(len_hc):
    schaefer_data.loc[len_rbd + k] = [FC_HC[k], ALFF_HC[k], ReHo_HC[j], fALFF_HC[j], 0]
~~~

k대신 j가 들어있었다.물론, 이게 사실이라면 코드가 안 돌아갔을 법하지만 그래도 확인차 다시 j를 k로 수정하고 다시 데이터 프레임을 만들어주려고 하였다.

데이터를 수정하고 나서 SVM으로 분류한 결과, 각 feature별 정확도는 아래와 같다.

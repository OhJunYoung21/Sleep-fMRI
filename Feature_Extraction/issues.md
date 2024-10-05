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

### CSV파일에서 데이터를 불러들일때는 문자열로 데이터를 읽어들인다.

Data Features(ALFF,Reho,Functional Connectivity)를 추출해서 csv 파일로 저장하였다.(매번 파이썬 코드를 실행해서 데이터를 얻어오는 과정이 귀찮았기 때문에)

csv파일에서 데이터를 읽어오려면 pd.read_csv()를 해야 하는데, 이때 데이터는 문자열 형태로 들어오게 된다.

문자열 형태로 들어온 데이터는 모델 트레이닝에 쓰일수 없기 때문에 numpy.ndarray형태또는 리스트 형태로 바꿔줘야 한다.

~~~python3
import ast
~~~

## 내가 겪은 오류들

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

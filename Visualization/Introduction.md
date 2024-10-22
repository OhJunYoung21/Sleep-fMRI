## Nilearn에서 제공하는 시각화 관련 함수들

### .inverse_transform()

매개변수로는 각 region별 signal을 받고, 결과물로는 .nii.gz와 같은 이미지 파일을 리턴한다.

~~~python3
shen_atlas = input_data.NiftiLabelsMasker(labels_img=atlas_path, standardize=True, strategy='mean',
                                          resampling_target="labels")

reho_rbd_data = shen_atlas.fit_transform(reho_img_rbd)
reho_hc_data = shen_atlas.fit_transform(reho_img_hc)

rbd_img_masked = shen_atlas.inverse_transform(reho_rbd_data)
hc_img_masked = shen_atlas.inverse_transform(reho_hc_data)
~~~

위 코드의 작동순서는 아래와 같다.

* shen_atlas라는 masker를 만들어준다.해당 Masker는 raw image에 shen_atlas를 씌워주는 역할을 하기 위해 미리 준비하는 과정이라고 생각하면 된다. 예를 들면 틀을 제작해준다고 생각하면 될 것이다.
* fit_transform을 사용해서 만든 틀을 실제 데이터에 적용해준다. 그 결과 각region에 맞는 signal들이 도출된다. node1 : {0.1231}, node2 : {0.6314}....
* inverse_transform을 통해 만들어진 signal을 이미지에 씌워준다. 위 이미지는 3D이미지가 된다.

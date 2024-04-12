## Find BOLD signal

PAR_REC 형태의 파일에서 BOLD signal을 찾는 코드는 아래와 같다.

~~~python3
import nibabel as nb
data_obj = nb.load('/Users/Desktop/Raw_data/RBD_fMRI/sub-01/ses-1/SHIN PARKINSON3_2_1.PAR')
data = data_obj.get_fdata()
print(data[100,100,10])
~~~

위 코드의 의미는 (100,100,10)좌표에 있는 voxel값의 BOLD signal을 표현한 것이다. 여기서 의문점이 생기는데, 왜 BOLD signal에 시간차원이 없냐는 것이다. 데이터가 3차원 구조인것으로 짐작하건대 해당 데이터는 시계열 데이터가 아닌 3차원 데이터인듯 싶다.

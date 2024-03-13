## PAR/REC to BIDS format

전달받은 데이터는 par/rec이지만, 이를 fmriprep-docker에 사용하려면 BIDS format으로 만들어줘야 한다.

### 사용가능한 도구들

* bidsify
* BIDScoin

### PAR/REC to NifTi file

~~~python3
import nibabel as nib
import os

par_path = '~/Desktop/RBD_fMRI/00021367 SHINKS PARKINSON3_3_1.PAR'
nifti_path = '~/Desktop/00021367 SHINKS PARKINSON3_3_1.nii.gz'

img = nib.load(par_path)
nifti = nib.Nifti1Image(img.dataobj, img.affine, header=img.header)
nifti.set_data_dtype('<f4')
nifti.to_filename(nifti_path)
~~~

위 코드를 사용하여 PAR/REC 형식의 파일을 nifti로 바꿔줄수 있다. 그러나 현재 내가 가진 PAR/REC은 90명이기 때문에 일일이 수행하는 것은 무리라고 생각된다.

### 자동화 작업

내가 다뤄야 하는 파일은 총 90개인데, 이를 일일이 입력하는 것은 비효율적이라고 생각하여 파일을 자동으로 NifTifh 만들어주는 코드를 작성하였다. 코드는 아래와 같다.

~~~python3
import nibabel as nib
import os

data_dir = '/Users/ojun-yong/Desktop/RBD_fMRI'


for filename in os.listdir(data_dir):
  if filename.endswith('.PAR'):
    par_path = os.path.join(data_dir, filename)
    (basename, ext) = os.path.splitext(filename)
    nifti_path = os.path.join(data_dir, f'{basename}.nii.gz')

    # PAR 파일 불러오기
    img = nib.load(par_path)

    # NIfTi 이미지 생성
    nifti = nib.Nifti1Image(img.dataobj, img.affine, header=img.header)
    nifti.set_data_dtype('<f4')

    # NIfTi 파일 저장
    nifti.to_filename(nifti_path)
~~~

실행해본 결과, 몇몇 파일은 제대로 전환되지 않았지마 일부는 작동하였다. 

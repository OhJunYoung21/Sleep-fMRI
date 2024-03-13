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

###BIDScoin

bidsify와 마찬가지로 PAR/REC 파일을 BIDS format으로 변형한다.

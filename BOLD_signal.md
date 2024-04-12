## Find BOLD signal

PAR_REC 형태의 파일에서 BOLD signal을 찾는 코드는 아래와 같다.

~~~python3
import nibabel as nb
data_obj = nb.load('/Users/Desktop/Raw_data/RBD_fMRI/sub-01/ses-1/SHIN PARKINSON3_2_1.PAR')
data = data_obj.get_fdata()
print(data[100,100,10])
~~~

단, 위 코드에서는 시간대별로 BOLD signal을 표현하지는 못한다. 시계열 데이터를 얻기 위해서는 Nifti파일로 변환해줘야 한다.

NIfti 파일로 변환해준 다음 해당 voxel의 전체 BOLD signal을 얻으려면 아래와 같이 코드를 작성하면 된다. 단순히 nb.load()안의 코드만 수정해주면 된다

~~~python3
data_obj = nb.load('/Users/ojun-yong/Desktop/bids_output/sub-01/ses-1/func/sub-01_ses-1_task_bold.nii')

data = data_obj.get_fdata()

print(data[100,100,10,0:10])
~~~

다음은 BOLD signal을 시각화 처리해보자.

~~~python

#아래 voxel의 BOLD signal을 보여줄 것이다.
data_show = data[100,100,10,:]

plt.plot(
    data_show,
    color='blue',
    marker='',
    linestyle='solid'
)
plt.xlim(0,100)
plt.ylim(200,800)
plt.title("BOLD signal")
plt.xlabel("Time")
plt.ylabel("Intensity")
plt.show()
~~~



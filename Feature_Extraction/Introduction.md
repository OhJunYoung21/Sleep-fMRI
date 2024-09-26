## Working Steps

1️⃣ 전처리된 데이터 불러오기

| RBD | HC | 
| ---| ---|
| 50개 | 39개 |

---

2️⃣ 전처리된 데이터에서 Confounds_regressed된 파일 추출하기

~~~python3
for index in range(len(fMRI_img)):
    # fMRI_image와 confounds 업로드

    fmri_img = fMRI_img[index]
    confounds = pd.read_csv(raw_confounds[index], sep='\t')

    confounds_of_interest = confounds[
        ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
         'global_signal', 'white_matter', 'csf']
    ]
    # 정규표현식을 사용해서 subject_number를 얻어온다. index+1로 얻어오는 경우, subject가 7,9인 경우 9를 8로 적어버리는 오류가 발생한다.
    subject_number = re.search(r'sub-(\d+)', fMRI_img[index]).group(1)

    cleaned_image = clean_img(fmri_img, confounds=confounds_of_interest)

    cleaned_image.to_filename(
        f"/Users/oj/Desktop/***_Lab/post_fMRI/confounds_regressed_HC/sub-{subject_number}_confounds_regressed.nii.gz")
~~~

---

3️⃣ Confounds_regressed된 파일들에서 alff,reho,fc 추출하기

해당 코드는 Shen_Features의 Classification_feature안에 들어있다.

---

4️⃣ 추출한 Feature들을 데이터 프레임에 넣어주고, RBD는 1로, HC는 0으로 라벨링하기

~~~python3
len_hc = len(ReHo_HC)
len_rbd = len(ReHo_RBD)

for j in range(len_hc):
    shen_data.loc[j] = [FC_HC[j], ALFF_HC[j], ReHo_HC[j], 0]

for k in range(len_rbd):
    shen_data.loc[len_hc + k] = [FC_RBD[k], ALFF_RBD[k], ReHo_RBD[k], 1]
~~~

마찬가지로, 자세한 코드는 Shen_features의 Classification_feature안에 들어있다.(자세한 설명이 필요하면 언제든지 comment 달아주세요.

import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA

# BOLD signal 데이터 로드
bold_signal = np.load('bold_signal.npy')  # 예시로 npy 파일을 사용했다고 가정합니다.

# 노이즈 데이터 로드
noise_data = pd.read_csv('noise.tsv', sep='\t', header=None)

# ICA 모델 생성
ica = FastICA(n_components=1)

# 노이즈 데이터에 ICA 적용
noise_components = ica.fit_transform(noise_data)

# BOLD 신호에서 노이즈 제거
cleaned_bold_signal = bold_signal - noise_components

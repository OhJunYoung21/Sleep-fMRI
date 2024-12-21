import pandas as pd
import os

shen_data = pd.read_pickle('shen_dynamic_final.pkl')

shen_data = shen_data.drop('prior_ALFF', axis=1)

# 268개의 지역에서 추출된 ALFF 값들의 variance로 예측을 하지만, 정상군과 환자군을 두고 봤을 때, 어떤 region(268개중)에서의 ALFF's variance 값이 /
# 두드러지게 차이가 나는지를 보는 것이 중요하다.

# 비단 내 데이터에서만 차이가 나는 것이 아닌, 다른 데이터를 적용하여도 해당 region의 ALFF's variance 값은 정상군과 환자군 사이에 차이가 있어야 한다.

'''
문제 접근 방식: 
1. 사전에 알려진 지식을 기반으로 feature를 추출한다.(ex. RBD 환자들은 특정 네트워크에서 ALFF,REHO,fALFF and FC 신호가 정상군과 다르다)
2. 분류작업을 실시하고, feature_importance
'''



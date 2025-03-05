#!/bin/bash


# pre_bids_folder안에는sub-01,sub02...등의 파일이 들어가 있다.
# after_bids_folder안에는 bidsmapper를 적용한 결과를 넣는다.

pre_bids_folder="$HOME/Desktop/pre_BIDS/pre_BIDS_RBD"
after_bids_folder="$HOME/Desktop/pre_BIDS/BIDS_RBD"


# 여기서 중요한 점: bidsmapper는 작업이 끝나고 나면 bidsmap.yaml을 저장하는데, 이는 반드시 after_bids_folder안에 저장해야 한다. 후에 bidscoiner를 사용할떄 사용할 예정이다. 쉽게 말하자면 bidsmap.yaml은 설계도 라고 생각하면 된다.

bidsmapper $pre_bids_folder $after_bids_folder
  
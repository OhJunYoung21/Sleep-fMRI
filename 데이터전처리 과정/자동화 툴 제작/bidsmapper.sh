#!/bin/bash


# pre_bids_folder안에는sub-01,sub02...등의 파일이 들어가 있다.
# after_bids_folder안에는 bidsmapper를 적용한 결과를 넣는다.

pre_bids_folder="$HOME/Desktop/pre_BIDS/pre_BIDS_RBD"
after_bids_folder="$HOME/Desktop/pre_BIDS/BIDS_RBD"

bidsmapper $pre_bids_folder @after_bids_folder
  
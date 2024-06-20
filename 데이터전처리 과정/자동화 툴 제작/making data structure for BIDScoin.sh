#!/bin/bash


folder_A="$HOME/Desktop/RAW_DATA/RBD/RBD_fMRI"

file_counts_A=$(ls -1q "$folder_A" | wc -l)

folder_B="$HOME/Desktop/RAW_DATA/RBD/RBD_T1"

file_counts_B=$(ls -1q "$folder_A" | wc -l)

target_folder="$HOME/Desktop/pre_BIDS/pre_BIDS_RBD"

##중간단계 확인

echo $file_counts_A
echo $file_counts_B

# pre_BIDS안에 sub-0*폴더 만들기

N=$((file_counts_A / 2))
echo $N

 
# pre_BIDS안에 파일 옮기기

files_fMRI=($(find "${folder_A}" -type f | sort))


echo ${#files_fMRI[@]}
  

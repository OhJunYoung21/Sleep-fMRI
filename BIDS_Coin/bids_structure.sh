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

for ((i = 1; i <= N; i++)); do
    folder_name=$(printf "sub-%02d" $i)
    mkdir -p "/$target_folder/$folder_name"
done

#fMRI안의 파일을 이름순으로 정렬하기

IFD=$'\n' files=($(ls "$folder_A" | sort))
unset IFS

#만들어진 폴더안에 파일 옮기기(PAR/REC)

for file in "$(ls -l "$folder_A" | sort)";do
	mv "$folder_A/$file" "target_folder/sub-01"
	done	


  

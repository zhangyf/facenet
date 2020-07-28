#!/bin/sh

for d in `ls $1`
do
	cur_path=$1$d
	if [ -d $cur_path ];then
		file_cnt=0
		for f in `ls $cur_path`
		do
			cur_file=$cur_path/$f
			if [ -f $cur_file ];then
				cur_files[$file_cnt]=$cur_file
        #        echo "put " $cur_file " at idx=" $file_cnt ${cur_files[$file_cnt]}
                ((file_cnt++))
			fi
		done
        
        start_idx=$(($RANDOM % $file_cnt))
        end_idx=$(($RANDOM % $file_cnt))
       
       if [ -f ${cur_files[$start_idx]} -a -f ${cur_files[$end_idx]} ];then
           for m in `ls models`
           do
               if [ -f "models/$m/model-$m.pb" ];then
                   cur_model="models/$m/model-$m.pb"
                   echo "python3 src/compare.py $cur_model ${cur_files[$start_idx]} ${cur_files[$end_idx]} > $m/$m-$d.out"
               fi
           done
           echo "================"
       fi
	fi
done



#python3 src/compare.py models/20200728-093152/model-20200728-093152.pb data/chinese_faces/faces_chinese_500_160/000/000_1.png data/chinese_faces/faces_chinese_500_160/000/000_4.png

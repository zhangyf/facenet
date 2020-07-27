#!/bin/sh

#nohup python3 -u src/train_softmax.py \
#	--logs_base_dir ~/workspace/facenet/ \
#	--models_base_dir ~/workspace/facenet/models/ \
#	--data_dir ~/workspace/facenet/data/chinese_faces/faces_chinese_500_160/ \
#	--image_size 160 \
#	--model_def models.inception_resnet_v1 \
#	--optimizer ADAM \
#	--learning_rate -1 \
#	--max_nrof_epochs 10 \
#	--epoch_size 500 \
#	--batch_size 90 \
#	--keep_probability 0.8 \
#	--random_crop \
#	--random_flip \
#	--use_fixed_image_standardization \
#	--learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt \
#	--weight_decay 5e-4 \
#	--embedding_size 512 \
#	--lfw_distance_metric 1 \
#	--lfw_use_flipped_images \
#	--lfw_subtract_mean \
#	--validation_set_split_ratio 0.05 \
#	--validate_every_n_epochs 5 \
#	--prelogits_norm_loss_factor 5e-4 1>nohup.out 2>&1 &

nohup python3 -u src/train_tripletloss.py \
	--logs_base_dir ~/workspace/facenet/ \
	--models_base_dir ~/workspace/facenet/models/ \
	--data_dir ~/workspace/facenet/data/chinese_faces/faces_chinese_500_160/ \
	--image_size 160 \
	--model_def models.inception_resnet_v1 \
	--optimizer ADAM \
	--learning_rate -1 \
	--max_nrof_epochs 10 \
	--epoch_size 500 \
	--batch_size 90 \
	--keep_probability 0.8 \
	--random_crop \
	--random_flip \
	--learning_rate_schedule_file data/learning_rate_retrain_tripletloss.txt \
	--weight_decay 5e-4 \
	--embedding_size 512 1>stdout.out 2>stderr.out &




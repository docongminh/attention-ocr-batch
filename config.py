# -*- coding: utf-8 -*-

import os
from parse_dict import get_dict

# base dir for multiple text datasets
# base_dir = '/home/quannd/Vin-BigData/Attention_OCR_Batch/dataset/imageout'
base_dir = '/u01/liemhd/dataset/reader/id/new/jpg'
base_dir = "dataclean_and_agm"
# font path for visualization
font_path = './fonts/cn/SourceHanSans-Normal.ttf'

# 'ocr' for inception model with padding image. 
# 'ocr_with_normalized_bbox' for inception model with cropped text region for attention lstm.
model_name = 'ocr' # 'ocr_with_normalized_bbox'

# path for tensorboard summary and checkpoint path 
# summary_path = './checkpoint'
# summary_path = './checkpoint_mobilenet_v1'
summary_path = './checkpoint_24x96_fake_and_real_clean_adadelta_continue_freeze_all_convlayer'

# tensorflow model name scope
# name_scope = 'MobilenetV1'
# name_scope = 'MobilenetV3'
name_scope = 'InceptionV4'
# path for numpy dict with processed image paths and labels used in dataset.py
dataset_name = ['image0.npy','image1.npy','image2.npy','image3.npy','image4.npy','image5.npy','image6.npy','image7.npy','image8.npy','image9.npy']
dataset_name = ['/home/dev/quand/inceptionv4/image0.npy',
 '/home/dev/quand/inceptionv4/image1.npy',
 '/home/dev/quand/inceptionv4/image2.npy',
 '/home/dev/quand/inceptionv4/image3.npy',
 '/home/dev/quand/inceptionv4/image4.npy',
 '/home/dev/quand/inceptionv4/image5.npy',
 '/home/dev/quand/inceptionv4/image6.npy',
 '/home/dev/quand/inceptionv4/image7.npy',
 '/home/dev/quand/inceptionv4/image8.npy',
 '/home/dev/quand/inceptionv4/image9.npy']

dataset_name = ['/home/dev/quand/inceptionv4/image0.npy']
dataset_name = ["image_12_3_2020_2m_0.npy","image_12_3_2020_2m_1.npy","image_12_3_2020_2m_2.npy","image_12_3_2020_2m_3.npy",
"image_12_3_2020_alldata_bgwhite_colorchar_0.npy",
"image_12_3_2020_alldata_bgwhite_colorchar_1.npy",
"image_12_3_2020_alldata_bgwhite_colorchar_2.npy",
"image_12_3_2020_alldata_bgwhite_colorchar_3.npy",
"image_12_3_2020_3m_0.npy",
"image_12_3_2020_3m_1.npy",
"image_12_3_2020_3m_2.npy",
"image_12_3_2020_3m_3.npy",
"image_12_3_2020_3m_4.npy",
"image_12_3_2020_3m_5.npy",
"data_1_4_2020_0.npy",
"data_1_4_2020_1.npy", 
"data_1_4_2020_2.npy", 
"data_1_4_2020_3.npy", 
"data_1_4_2020_4.npy", 
"data_1_4_2020_5.npy", 
]
#dataset_name = ["data_1_4_2020_5.npy"]
#dataset_name = ["data_1_4_2020_realdata_aliem_resize_0.npy"]
dataset_name = [ "data_10_4_2020_dataclean_and_agm_0.npy",
"data_1_4_2020_1.npy",
"image_12_3_2020_2m_0.npy",
"image_12_3_2020_2m_1.npy",
"image_12_3_2020_3m_0.npy",
"image_12_3_2020_3m_2.npy",
"image_12_3_2020_alldata_bgwhite_colorchar_0.npy",
"data_10_4_2020_dataclean_and_agm_1.npy",
"data_10_4_2020_dataclean_and_agm_2.npy",
"data_10_4_2020_dataclean_and_agm_3.npy",
"data_10_4_2020_dataclean_and_agm_4.npy",
]
#dataset_name = [ "data_10_4_2020_dataclean_and_agm_0.npy"] 
# pb_path = './checkpoint/text_recognition_5435.pb'

# restore training parameters
restore_path = ''
starting_epoch = 0

# checkpoint_path = './checkpoint/model-10000'
# imagenet pretrain model path
pretrain_path = 'checkpoint_24x96_only_fake_data_continue/model-324000.data-00000-of-00001' 
#pretrain_path = "checkpoint_24x96_fake_and_real_adadelta_best_444000/model-444000.data-00000-of-00001"
# pretrain_path = '/home/quannd/Vin-BigData/inceptionv4/inceptionv4/checkpoint_32*128/model-148200'
# pretrain_path = '/home/quannd/Vin-BigData/Attention_OCR_Batch/pretrain/v3-large_224_1.0_float/pristine/model.ckpt-000001.data-00000-of-00001'
# pretrain_path = '/home/quannd/Vin-BigData/Attention_OCR_Batch/pretrain/v3-large_224_1.0_float/epoch4/model.ckpt-549765.data-00000-of-00001'
# pretrain_path = '/home/quannd/Vin-BigData/Attention_OCR_Batch/pretrain/mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.ckpt.data-00000-of-00001'
# pretrain_path = '/home/quannd/Vin-BigData/inceptionv4/Attention_OCR_Batch/inception_v4_2016_09_09/inception_v4.ckpt'
# label dict for text recognition
label_dict = get_dict()
reverse_label_dict = dict((v,k) for k,v in label_dict.items())

# gpu lists
gpus = [1]

num_gpus = len(gpus)
num_classes = len(label_dict)

# max sequence length without EOS 
seq_len = 32

# embedding size
wemb_size = 256

# lstm size
lstm_size = 256

# minimum cropped image size for data augment
crop_min_size = 22
crop_min_size_width = 84
# input image size
image_size = 24
image_size_width = 96

# max random image offset for data augment
offset = 16

# CNN endpoint stride
stride = 4

# resize parameters for data augment
TRAIN_SHORT_EDGE_SIZE = 4
MAX_SIZE = image_size - 8

# training batch size
batch_size = 512 #12

# steps per training epoch in tensorpack
steps_per_epoch = 10000

# max epochs 
num_epochs = 50

# model weight decay factor
weight_decay = 1e-5

# base learning rate
learning_rate = 0.001 * num_gpus

# minimun learning rate for cosine decay learning rate
min_lr = learning_rate / 100

# warm up steps 
warmup_steps = 10000

# thread for multi-thread data loading  
num_threads =64

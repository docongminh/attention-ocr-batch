#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import cv2
import time
import argparse
import numpy as np
#from matplotlib import pyplot as plt

import config as cfg
from common import polygons_to_mask

from model.tensorpack_model import *

from tensorpack.predict import MultiTowerOfflinePredictor, OfflinePredictor, PredictConfig, MultiThreadAsyncPredictor,MultiTowerOfflinePredictor
from tensorpack.tfutils import SmartInit, get_tf_version_tuple
from tensorpack.tfutils.export import ModelExporter
def check(a):
    for i in a:
        if not i.isalpha():
            return False
    return True

def cal_sim(str1, str2):
    """
    Normalized Edit Distance metric (1-N.E.D specifically)
    """
    m = len(str1) + 1
    n = len(str2) + 1
    matrix = np.zeros((m, n))
    for i in range(m):
        matrix[i][0] = i
        
    for j in range(n):
        matrix[0][j] = j

    for i in range(1, m):
        for j in range(1, n):
            if str1[i - 1] == str2[j - 1]:
                matrix[i][j] = matrix[i - 1][j - 1]
            else:
                matrix[i][j] = min(matrix[i - 1][j - 1], min(matrix[i][j - 1], matrix[i - 1][j])) + 1
    
    lev = matrix[m-1][n-1]
    if (max(m-1,n-1)) == 0:
        sim = 1.0
    else:
        sim = 1.0-lev/(max(m-1,n-1))
    return sim

import copy
def preprocess(image, points, size=cfg.image_size,size_width = cfg.image_size_width):
    """
    Preprocess for test.
    Args:
        image: test image
        points: text polygon
        size: test image size
    """
    height, width = image.shape[:2]
    w = width
    h = height
    # x = 0
    # y = 0 
    # w = width - 1
    # h = height - 1
    # image = image[y:y+h, x:x+w,:]
    # new_height, new_width = (size, int(w*size/h)) if h>w else (int(h*size/w), size)
    new_height = size 
    new_width = min(size_width,int(w*size/h))
    image = cv2.resize(image, (new_width, new_height))
    # print(image)
    # if new_height > new_width:
    padding_top, padding_down = 0, 0
    padding_left = (size_width - new_width)//2
    padding_right = size_width - padding_left - new_width
    # else:
    #     padding_left, padding_right = 0, 0
    #     padding_top = (size - new_height)//2
    #     padding_down = size - padding_top - new_height

    image = cv2.copyMakeBorder(image, padding_top, padding_down, padding_left, padding_right, borderType=cv2.BORDER_CONSTANT, value=[255,255,255])
    # print('shape img' + str(image.shape))
    # cv2.imwrite('quan.jpg',image)
    image = image/255.
    return image


def label2str(preds, probs, label_dict, eos='EOS'):
    """
    Predicted sequence to string. 
    """
    results = []
    for idx in preds:
        if label_dict[idx] == eos:
            break
        results.append(label_dict[idx])

    probabilities = probs[:min(len(results)+1, cfg.seq_len+1)]
    return ''.join(results), probabilities

# def eval(args, filenames, polygons, labels, label_dict=cfg.label_dict):

def check_character_level(gt_string, pred_string):
    gt_string = gt_string
    pred_string = pred_string
    s = 0
    for i in range(len(gt_string)):
        # if gt_string[i] == '*' and i > 0:
        #     break
        if gt_string[i] == pred_string[i]:
            s+= 1
    return s/i

def cal_accuracy(GT, PRED, files, file_log):
    word_acc = 0
    char_acc = 0
    for i, gt in enumerate(GT):
        gt_string = gt
        pred_string = PRED
        char_acc += check_character_level(gt_string, pred_string)
        if gt_string == pred_string:
            word_acc += 1
        else:
            with open(file_log, "a+") as f:
                str_w = files[i] + " " + gt_string + " " + pred_string
                f.write(str_w)
                f.write("\n")
    return word_acc/GT.shape[0], char_acc/GT.shape[0]
import os
def loaibokytuthua(s) :
    for s1 in ":,.- ()" :
        while (s.find(s1)!=-1) :
            s = s.replace(s1,"")
    return s

def unpad(height) :
    return max(min(height/5,4),6)+height+max(min(height/10,2),4)

def getheight(h) :
#     a = []
    for i in range(h):
        if unpad(i)==h:
            return i



def getOriginShape(img):
    height, width = img.shape[:2]
    h = getheight(height)
    padtop = max(min(h/5,4),6)
    padother = max(min(h/10,2),4)
    img = img[padtop:height-padother,padother:width-padother,:]
    # return img
    return padtop,height-padother,padother,width-padother

def check(a):
    for i in a:
        if not i.isalpha():
            return False
    return True

def eval(args, file_label, img_folder ,label_dict=cfg.label_dict,infer = 0):
    Normalized_ED = 0.
    total_num = 0
    total_time = 0

    model = AttentionOCR()
    predcfg = PredictConfig(
        model=model,
        session_init=SmartInit(args.checkpoint_path),
        input_names=model.get_inferene_tensor_names()[0],
        output_names=model.get_inferene_tensor_names()[1])

    predictor = OfflinePredictor(predcfg)
    # predictor = MultiThreadAsyncPredictor(predcfg, batch_size= 5)
    if infer ==1 :
        open('kq.txt','w').close()
        f1 = open('kq.txt','w')
        # f1 = open('realdata.txt','r',encoding='utf8')
        # lf = os.listdir('/home/quannd/Vin-BigData/inceptionv4/inceptionv4/test_bc (2)/test_bc/all')
        # lf = os.listdir('/home/quannd/Vin-BigData/Attention_OCR_Batch/business_card_example (2)/test')
        lf = os.listdir('/home/quannd/Vin-BigData/inceptionv4/inceptionv4/wrong_reader_32x128/BC_read_1579501317390')
        print(len(lf))
        dem = 0
        c = 0
        for f in sorted(lf):
            # f, lb = line.split('\t')
            # lb = lb.strip() 
            # image = cv2.imread(os.path.join('/home/quannd/Vin-BigData/inceptionv4/inceptionv4/test_bc (2)/test_bc/all',f))
            image = cv2.imread(os.path.join('/home/quannd/Vin-BigData/inceptionv4/inceptionv4/wrong_reader_32x128/BC_read_1579501317390',f))
            if image is None :
                print(f)
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
            # # print(image.shape)
            image_color = np.zeros([image.shape[0],image.shape[1],3],dtype=np.uint8)
            image_color[:,:,0]=image
            image_color[:,:,1]=image
            image_color[:,:,2]=image
            image = image_color
            # image = image[3:image]
            height, width = image.shape[:2]
            image = image[1:height-1,1:width-1,:]
            image = 255 - image
            k1= 3
            k2 = 1
            kernel1 = np.ones((k1,k1),np.uint8)
            kernel2 = np.ones((k2,k2),np.uint8)
            image = cv2.erode(image,kernel1,iterations = 1)
            image = cv2.dilate(image,kernel2,iterations = 1)
            cv2.imwrite('erode/'+f,image)
            height, width = image.shape[:2]
            points = [[0,0], [width-1,0], [width-1,height-1], [0,height-1]]
            image = preprocess(image, points, cfg.image_size)
            preds, probs = predictor(np.expand_dims(image, axis=0), np.ones([1,cfg.seq_len+1], np.int32), False, 1.)
            preds, probs = label2str(preds[0], probs[0], label_dict)
            print(preds)
            # print(lb)
            # dem += 1
            # if preds.upper()==lb.upper():
            #     c+=1
            # else:
            #     print(preds)
            #     print(lb)
            f1.write(f+'\t'+preds+'\n')
        f1.close()
        # print(dem)
        # print('kq~~~~~~~~~~~~~~~~~~~',c/dem)
        return 

    count = 0 
    if infer==2 :
        open('kq.txt','w').close()
        f2 = open('kq.txt','w')
        f1 = open('allrealdata.txt','r',encoding='utf8')
        # f1 = open('/u01/liemhd/dataset/reader/id/new/train_v3.txt','r',encoding='utf8') 
        # lf = os.listdir('/home/quannd/Vin-BigData/inceptionv4/inceptionv4/test_bc (2)/test_bc/all')
        #lf = os.listdir('/home/quannd/Vin-BigData/Attention_OCR_Batch/business_card_example (2)/test')
        # print(len(lf))
        dem = 0
        c = 0
        for line in f1.readlines():
            f, lb = line.split('\t')
            lb = lb.strip() 
            ck = False
            for l in lb:
                if not l.isalpha():
                     ck = True
            if ck : 
                continue
            # if not path.exists(os.path.join('/home/quannd/Vin-BigData/inceptionv4/inceptionv4/test_bc (2)/test_bc/all_new',f)) 
            image = cv2.imread(os.path.join('allrealdata',f))
            # image = cv2.imread(os.path.join('/home/quannd/Vin-BigData/Attention_OCR_Batch/business_card_example (2)/test',f))
            if image is None :
                print(f)
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
            # # # # print(image.shape)
            # image_color = np.zeros([image.shape[0],image.shape[1],3],dtype=np.uint8)
            # image_color[:,:,0]=image
            # image_color[:,:,1]=image
            # image_color[:,:,2]=image
            # image = image_color
            # kernel = np.ones((3,3),np.uint8)
            # image = cv2.erode(image,kernel,iterations = 1)
            # image = image[3:image]
            height, width = image.shape[:2]
            # padall = 6
            # padtop = max(min(int((height-12)/10),5),3)
            # padother = max(min(int((height-12)/10),5),3)
            # image = image[padall-padtop:height-padall+padother,padall-padother:width-padall+padother,:]
            # image = image[4:height-2,2:width-2,:]

            x1,x2,y1,y2 = getOriginShape(image)
            h = x2-x1
            padtop = max(min(int(h*2/100),6),4)
            padother = max(min(int(h*2/200),4),2)
            if (x1-padtop<0) or (x2+padtop>height) or (y1-padother<0) or (y2+padother>width):
                count+=1
            image = image[max(0,x1-padtop):min(height,x2+padtop),max(0,y1-padother):min(width,y2+padother),:]
            height, width = image.shape[:2]
            points = [[0,0], [width-1,0], [width-1,height-1], [0,height-1]]
            image = preprocess(image, points, cfg.image_size)
            preds, probs = predictor(np.expand_dims(image, axis=0), np.ones([1,cfg.seq_len+1], np.int32), False, 1.)
            preds, probs = label2str(preds[0], probs[0], label_dict)
            # print(preds)
            # print(lb)
            preds = loaibokytuthua(preds)
            lb = loaibokytuthua(lb)
            # preds = preds.strip(":,.-").upper()
            # lb = lb.strip(":,.-").upper()
            nhan = b'\\u0110'.decode('unicode_escape')#nhan
            predict = b'\\xd0'.decode('unicode_escape')#predict
            lb = lb.replace(nhan,predict)
            preds = preds.replace(nhan,predict)
            dem += 1
            check = "dung"
            if (preds=="" ) :
                continue
            if preds.upper()==lb.upper():
                c+=1
            else:
                print(f)
                print(preds)
                print(lb)
                check = "sai"
                # print(padother)
                # print(padtop)
            f2.write(f+'\t'+preds.upper()+'\t'+lb.upper()+'\t'+check+'\n')
        print(count)
        f1.close()
        f2.close()
        print(dem)
        print('kq~~~~~~~~~~~~~~~~~~~',c/dem)
        return 

    with open(file_label, 'r',encoding='utf-8') as f:
        lines = f.readlines()
        # print(lines)
        word_acc = 0
        char_acc = 0
        file_log='log_result.txt'
        for line in lines[:2000]:
            file, label = line.strip().split('\t', 1)
            filename = os.path.join(img_folder, file)
    # for filename, points, label in zip(filenames, polygons, labels):
            try:
                image = cv2.imread(filename)
                # print(image)
                if image is None :
                    print(filename)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                height, width = image.shape[:2]
                points = [[0,0], [width-1,0], [width-1,height-1], [0,height-1]]
                image = preprocess(image, points, cfg.image_size)

                before = time.time()
                preds, probs = predictor(np.expand_dims(image, axis=0), np.ones([1,cfg.seq_len+1], np.int32), False, 1.)
                after = time.time()

                total_time += after - before
                # print(preds)
                preds, probs = label2str(preds[0], probs[0], label_dict)
                # print(label)
                # print(preds, probs)
                # char_acc += check_character_level(label, preds)
                if label.upper() == preds.upper():
                    word_acc += 1
                    # print(label)
                    # print(preds, probs)                   
                else:
                #     with open(file_log, "a+") as f:
                #         str_w = filename + " " + label + " " + preds
                #         f.write(str_w)
                #         f.write("\n")
                    print(label)
                    print(preds, probs)     
                    # print(cal_sim(preds, label))  
                # sim = cal_sim(preds, label)
                sim = 0

                total_num += 1
                Normalized_ED += sim
            except Exception as e:
                print(e)
                print(e.with_traceback())
                continue
        print("Accuracy")
        print(word_acc/total_num)
        print(char_acc/total_num)

    print("total_num: %d, 1-N.E.D: %.4f, average time: %.4f" % (total_num, Normalized_ED/total_num, total_time/total_num))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OCR')
    modelfolder= 'checkpoint_24x96_fake_and_real_clean_adadelta_continue_freeze_all_convlayer'
    f=open(modelfolder+'/checkpoint','r')
    l=''
    for x in f.readlines():
        l=x
        break
    ln = len('model_checkpoint_path: ')
    model_name = modelfolder+'/'+l[ln:].strip().strip('"')
    # model_name = modelfolder+'/' + 'model-81900'
    # model_name = 'checkpoint_32x100 (copy)/model-163800'
    f.close()
    # model_name = "checkpoint_20x80/model-31248" 8850
    # checkpoint_20x80/model-62496 884
    # checkpoint_20x80/model-78120 894
    # checkpoint_20x80/model-140616 91
    # model_name = "checkpoint_20x80/model-78120"
    # model_name = 'checkpoint_32x100/model-358800' 8449
    # model_name = 'checkpoint_32x100/model-358800' 
    # model_name = 'checkpoint_32x100/model-358800' 8449
    # model_name = 'checkpoint_32x100/model-358800' 8449
    # model_name = 'checkpoint_32x100/model-358800' 8449
    #model_name = 'checkpoint_24x96_fake_and_real_clean_adadelta_continue/model-404000' 
    model_name = "checkpoint_24x96_fake_and_real_clean_adadelta_continue_freeze_all_convlayer/model-624000"
    #model_name = "checkpoint_24x96_only_fake_data_continue/model-198000"
    parser.add_argument('--checkpoint_path', type=str, help='path to tensorflow model', default=''+model_name)
    # parser.add_argument('--checkpoint_path', type=str, help='path to tensorflow model', default='/home/quannd/Vin-BigData/inceptionv4/checkpoint_inceptionv4/checkpoint_inceptionv4/model-280248')
    args = parser.parse_args()

    from dataset import ICDAR2017RCTW

    # ICDAR2017RCTW = ICDAR2017RCTW()
    # ICDAR2017RCTW.load_data()
    # print(len(ICDAR2017RCTW.filenames))

    # file = '/home/quannd/Vin-BigData/gen_businesscard/all/image/allimage.txt'
    # img_folder='/home/quannd/Vin-BigData/gen_businesscard/all/image/allimage'
    
    file = '/home/quannd/Vin-BigData/gen_businesscard/alldata_3_3_2020/test.txt'
    img_folder='/home/quannd/Vin-BigData/gen_businesscard/alldata_3_3_2020/test'
    eval(args, file, img_folder, infer = 2)
    print(model_name)

    # eval(args, ICDAR2017RCTW.filenames, ICDAR2017RCTW.points, ICDAR2017RCTW.transcripts)

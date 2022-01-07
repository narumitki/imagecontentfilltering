import time
import numpy as np
import datetime
import cv2
from flask import Flask, request

from yolov3.utils import detect_image_API
from yolov3.yolov4 import Create_Yolo
from yolov3.configs import *
from tensorflow.keras.models import load_model

from Content_model import content_filtering
from config import *

#Call model
#Content_Classification_model
Content_Classification_model = load_model(CNN_model_path)
#YOLO
yolo = Create_Yolo(input_size = input_size, CLASSES=TRAIN_CLASSES)
yolo.load_weights(Yolo_weight_path) # use keras weights

def upload_file(file_name):
    Return_content = {'classPredict':[],'imageName':[],'irregularClass':[],'timestamp':[]}
    try:
        if {file_name[-4:].lower()}.issubset(set(fileExtensionAllowed)):
            start_time = time.time()
            #Read image
            image = cv2.imread(file_name)
            #Predict with yolo
            start_time = time.time()
            image,bbox_class_list = detect_image_API(yolo, image, "", input_size=input_size, show=False, CLASSES=TRAIN_CLASSES, score_threshold=score_threshold, iou_threshold=iou_threshold, rectangle_colors=(255,0,0)) #yolo
            print('Yolo--->',bbox_class_list)
            print('-----------------------------------------------------------------------------------------------------------')
            start_time = time.time()
            if bbox_class_list != []:
                #Predict with CNN
                content,irregularClass = content_filtering(image,bbox_class_list,Content_Classification_model,interest_content,irregular_content,Class_detect)#2sec CNN
                #cv2.imwrite(file_name[:-4]+'_bbox.jpg',image)
                
                Return_content['classPredict'].append(content)
                Return_content['imageName'].append(file_name)
                Return_content['irregularClass'].append(irregularClass)
                Return_content['timestamp'].append(datetime.datetime.now())
                runtime = "--- %s seconds ---" % (time.time() - start_time)
                print(content)
                print(runtime)
            else:

                Return_content['classPredict'].append(Class_detect)
                Return_content['imageName'].append(file_name)
                Return_content['irregularClass'].append("Regular")
                Return_content['timestamp'].append(datetime.datetime.now())
        else:
            return "Error: Please re-upload img."
        print('-----------------------------------------------------------------------------------------------------------')
    except Exception as e:
        print(e)
        return "Error: Please re-upload img."


#-----------------------------------------------------------------------------------------------------------
file_name = '3.jpg'
upload_file(file_name)
    


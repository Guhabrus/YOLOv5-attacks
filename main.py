

from audioop import add
from cgi import print_form
from inspect import trace
import os
import shutil
import numpy as np

import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from Geo import get_coords


PATH = "/home/maksim/Myfolder/Magistr/VKR/Раскадровка/"



name = "zvezda" 
date = "_"
part = ""
foramat = ".MP4"
cap = cv2.VideoCapture(name+foramat)

COUNT2 = 0


count = 0

def get_shot():

    if cap.isOpened() == False:
        print('Не возможно открыть файл')
    print(count)       
    while cap.isOpened():
 
        fl, img = cap.read()

        if img is None:
            break

        print(count)
        if (count % 50 == 0):

            if 1  :
                img = img[70:img.shape[0] ,:img.shape[1]-200]
                cv2.imwrite(str(count)+date + name+part + ".png", img)
            count+=1
            
            
        print(img.shape[0], img.shape[1])
    cap.release()
    # закрываем все открытые opencv окна
    cv2.destroyAllWindows()




def work():
    for root, dirs, files in os.walk(PATH):
        for filename in files:
            print(PATH +filename)
            if(PATH +filename == PATH + "main.py") or (PATH +filename == (PATH + name+foramat)):
                continue
            
            image = cv2.imread(PATH +filename)   

            cv2.imshow("Original image", image)
            
            key = cv2.waitKey(0) 
            if key == ord('n'):
                os.replace(PATH+filename, PATH+"as/"+filename)
                continue    
            elif key == ord('d'):
                os.remove(PATH +filename)
            elif key == ord('q'):
                break
        # break
            
def get_obj_img(A:list,B:list):
    '''
    [in] А - координата верхнего левого угла
    [in] B - координата нижнего правого угла
    [out] - координаты середины точек квадрата
    '''
    print(f"A - {A}, B - {B}")
    x = A[0] + (B[0]-A[0])/2
    y = A[1] + (B[1]-A[1])/2
    # x2 = [ A[0] + (B[0]- A[0]) , A[1]]
    # x3 = B
    # x4 = [A[0], A[1] + (B[1] - A[1])]
    print(f"x1 = {x} x2 = {y}")
    return x,y


    

def detect_batch(source, model: TracedModel, dataset:LoadImages, colors:list, names:list, imgsz = 640 ,device = select_device("") , trace = True):

    result_batch = []
    result_coord = []

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    
    for path, img,  im0s, vid_cap in dataset:
        
        img = torch.from_numpy(img).to(device)
        img = img.half() if ( device.type != 'cpu') else img.float()  
        img /= 255.0  
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

                
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=False)[0]

        with torch.no_grad():  
            pred = model(img, augment=False)[0]
            
        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic = False)

        for i, det in enumerate(pred):  
          
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
 
            if len(det):
                
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  
                count = 0
                
                for *xyxy, conf, cls in reversed(det):   
                    print(f"\n\nxyxy - {xyxy}, conf - {conf} class - {cls}")

                    x,y = get_obj_img( [float(xyxy[0]), float(xyxy[1])], [float(xyxy[2]),float(xyxy[3])] )
                    geo_crds = get_coords(x,y)
                    print(f"geo coords - {geo_crds}")
                    label = f'{names[int(cls)]} {conf:.2f} {geo_crds}'
                    print(f"label - {label}")
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                    # img_s = im0[int(xyxy[1]) :int(xyxy[3]), int(xyxy[0]) : int(xyxy[2]) ]
                    
                    result_coord.append( [geo_crds])
                    result_batch.append(im0)

                    
            
            if dataset.mode == 'image':
                return result_coord, result_batch

            
        

    print('Done.)')


def detect():
    trace = True

    device = select_device("")

    weights_path = "weights/epoch_624.pt"
    img_size = 640
    img_source = "/home/maksim/Myfolder/Magistr/VKR/Раскадровка/parad.jpg"

   #TODO получить размер изображения

    model = attempt_load(weights_path, map_location=device)  


    stride = int(model.stride.max())  
    imgsz = check_img_size(img_size, s=stride)  

    if trace:
        model = TracedModel(model, device, img_size)  

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    dataset = LoadImages(img_source, img_size=imgsz, stride=stride)

    res_coord, result_batch = detect_batch(img_source, model, dataset, colors, names, device=device, imgsz = img_size)
    print(f"res- {res_coord}")
    for i in result_batch:
        cv2.imshow("Img", i)
        cv2.imwrite("rest.jpg", i)
        key = cv2.waitKey(0) 
        if key == ord('q'):
            break
    
if __name__ == '__main__':
    
    detect()




# get_shot()



import os
import shutil


import argparse
import time
from pathlib import Path

import cv2

from numpy import random


PATH = "/home/maksim/Myfolder/Magistr/VKR/Раскадровка/Data/"



name = "zvezda" 
date = "_"
part = ""
foramat = ".MP4"
cap = cv2.VideoCapture(PATH+name+foramat)

COUNT2 = 0




def get_shot(count = 0):

    if cap.isOpened() == False:
        print('Не возможно открыть файл')
    # print(count)       

    while cap.isOpened():
 
        fl, img = cap.read()

        if img is None:
            break

        
        cv2.imshow("Cat", img)
        print(img.shape[0], img.shape[1])
        key = cv2.waitKey(0) 
        
        if key== ord('q'):
            break
        if key == ord('n'):
            cv2.imwrite(str(count)+date + name+part +str(count)+ ".png", img)
        if key == ord('d'):
            ...
        
        # if (count % 54 == 0):

        #     if 1  :
        #         img = img[:img.shape[0] ,:img.shape[1]]
                
        #         cv2.imwrite(str(count)+date + name+part +str(count)+ ".png", img)
        #         # break
        count+=1
        
                # COUNT2+=1
                # break
        
            
        
    cap.release()
    # закрываем все открытые opencv окна
    cv2.destroyAllWindows()




def work(path_out = "dev_"):
 
    if(not os.path.exists(path_out)):
        os.mkdir(path_out)
    for root, dirs, files in os.walk(PATH):
        files.sort()
        for filename in files:
            print(PATH +filename)

            
            print(files)
            
            if(PATH +filename == PATH + "pr_data.py") or (PATH +filename == (PATH + name+foramat)):
                continue
            
            image = cv2.imread(PATH +filename)   
            
            cv2.imshow("Original image", image)
            
            key = cv2.waitKey(0) 
            if key == ord('n'):
                os.replace(PATH+filename, PATH+"dev_/"+filename)
                continue    
            elif key == ord('d'):
                os.remove(PATH +filename)
            elif key == ord('q'):
                break
        break
            


get_shot()

# work()

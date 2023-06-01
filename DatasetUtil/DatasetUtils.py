from importlib.resources import path
import os
import cv2


PATH = "/home/maksim/Myfolder/Magistr/VKR/Раскадровка/RebuildDataset/test.v3i.yolov7pytorch/train/images/"

def delete_from_dataset(name = ""):
    if(name == ""):
        print("Name - ????")
        exit()

    path_img = "/home/maksim/Myfolder/Magistr/VKR/Раскадровка/RebuildDataset/test.v3i.yolov7pytorch/train/images/" + name
    path_txt = "/home/maksim/Myfolder/Magistr/VKR/Раскадровка/RebuildDataset/test.v3i.yolov7pytorch/train/labels/" + name[:-4] + ".txt"

    
    if(os.path.exists(path_img)):
        os.remove(path_img)
    else:
        print("dont delete ----" + path_img)
   

    if(os.path.exists(path_txt)):
        os.remove(path_txt)
    else:
        print("dont delete ----" + path_txt)
    




def work():
    for root, dirs, files in os.walk(PATH):
        for filename in files:
            print(filename)
         
            
            image = cv2.imread(PATH +filename)   

            cv2.imshow("Original image", image)
            
            key = cv2.waitKey(0) 
            if key == ord('n'):
                os.replace(PATH+filename, PATH+"as/"+filename)
                continue    
            elif key == ord('d'):
                delete_from_dataset(filename)
            elif key == ord('q'):
                break
        # break


work()
import cv2
import os

cascade_path =  "./cascades/haarcascade_frontalface_default.xml"
img_path = "./images/"
out_path = "./trimmed/"

files = os.listdir(img_path)
cascade = cv2.CascadeClassifier(cascade_path)

for file in files:
    src = cv2.imread(img_path+file,0)
    gray = cv2.cvtColor(src,cv2.cv2.COLOR_BAYER_BG2GRAY)
    rect = cascade.detectMultiScale(gray)
    if len(rect) > 0:
        for i,[x, y, w, h] in enumerate(rect):
            img_trimmed = src[y:y + h, x:x + w]
            file_name = "{}_{}".format(i,file)
            file_path = out_path + file_name
            cv2.imwrite(file_path, img_trimmed)

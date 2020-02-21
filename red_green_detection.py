import cv2 as cv
import numpy as np
import os
import natsort


def show(img):
    cv.imshow("", img)
    cv.waitKey()
    cv.destroyAllWindows()

def color_detect(frame):
        hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)  
        
        lower_green = np.array([30,30,30])             
        upper_green = np.array([90,255,255])

        lower_red = np.array([0, 50, 50])             
        upper_red = np.array([10, 255, 255])

        mask_GREEN = cv.inRange(hsv,lower_green,upper_green)
        mask_RED = cv.inRange(hsv,lower_red,upper_red)

        GREEN_RESULT = cv.bitwise_and(frame, frame, mask = mask_GREEN)
        RED_RESULT = cv.bitwise_and(frame, frame, mask = mask_RED)

        cv.imshow('frame',frame)
        cv.imshow('GREEN',GREEN_RESULT)
        cv.imshow('RED',RED_RESULT)
        cv.waitKey()
        cv.destroyAllWindows()


def main():
    # 전체 이미지 목록받고, 오름차순 정렬
    Gimg_list = os.listdir("./img/green/")
    Rimg_list = os.listdir("./img/red/")
    
    Gimg_list = natsort.natsorted(Gimg_list)
    Rimg_list = natsort.natsorted(Rimg_list)

    for g in Gimg_list:
        img = cv.imread(f"./img/green/{g}")
        show(img)
        color_detect(img)
    
    for r in Rimg_list:
        img = cv.imread(f"./img/red/{r}")
        show(img)
        color_detect(img)
        
main()
import cv2 as cv
import numpy as np
import os
import sys
import natsort

# 배열 전체 출력을 위해서
np.set_printoptions(threshold=sys.maxsize) 

def show(img):
    cv.imshow("", img)
    cv.waitKey()
    cv.destroyAllWindows()


# 아래 2줄의 init_img와 G_copy는 단순히 초기화를 위한 용도
init_img = np.zeros((1,1))
init_copy = init_img.copy()

# mouse event를 위한 변수들 
moues_pressed = False
s_x = s_y  = e_x = e_y = -1
def mouse_callback(event, x, y, flags, param):
    global init_img, init_copy, s_x, s_y, e_x, e_y, moues_pressed
    if event == cv.EVENT_LBUTTONDOWN:
        moues_pressed = True
        s_x, s_y = x, y
        init_img = init_img.copy()
        
    elif event == cv.EVENT_MOUSEMOVE:
        if moues_pressed:
            init_copy = init_img.copy()
            cv.rectangle(init_copy, (s_x, s_y), (x, y), (0, 255, 0), 3)
    
    elif event == cv.EVENT_LBUTTONUP:
        moues_pressed = False
        e_x, e_y = x, y


def red_detection(frame):
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)  
        
    lower_red = np.array([0, 50, 50])             
    upper_red = np.array([10, 255, 255])
    
    mask_RED = cv.inRange(hsv,lower_red,upper_red)    
    RED_RESULT = cv.bitwise_and(frame, frame, mask = mask_RED)
    
    return RED_RESULT
    
    
def init(img):
    '''
    첫 번째 이미지에서 신호등만 마우스로 드래그
    '''
    global init_img, init_copy, s_x, s_y, e_x, e_y, moues_pressed
    w, h = img.shape[0], img.shape[1]
    init_copy = img.copy()
    init_img = img.copy()
    history = img.copy()
    
    min_point_list = []
    max_point_list = []
    

    cv.namedWindow("DRAG Traffic Light and Press 'w' !")
    cv.setMouseCallback("DRAG Traffic Light and Press 'w' !", mouse_callback)    
    while True:
        cv.imshow("DRAG Traffic Light and Press 'w' !", init_copy)
        key = cv.waitKey(1)

        if key == ord('w'):
            if s_y > e_y:
                s_y, e_y = e_y, s_y
            if s_x > e_x:
                s_x , e_x = e_x, s_x

            if e_y - s_y > 1 and e_x - s_x > 0:
                cv.rectangle(init_copy, (s_x, s_y), (e_x, e_y), (0, 0, 255), 3)
                min_point_list.append([s_x, s_y])
                max_point_list.append([e_x, e_y])
                break
        
        # 박스 그리기 실수 했을 경우 esc누르면 다시 시작.
        elif key == 27:
            init_copy = init_img.copy()
            continue
        
    cv.destroyAllWindows()
    
    return min_point_list, max_point_list, init_copy


def rotate_img(src):
    height, width, channel = src.shape
    matrix = cv.getRotationMatrix2D((width/2, height/2), -90, 1)
    dst = cv.warpAffine(src, matrix, (width, height))

    return dst


def main():
    global init_img, init_copy

    video_path = '/home/junsoofeb/py_project/traffic_light_detection/traffic_light_1.mp4'
    cap = cv.VideoCapture(video_path)

    _, first_img = cap.read()
    first_img = rotate_img(first_img)
    H, W, _ = first_img.shape
    min_point_list, max_point_list, _ = init(first_img.copy())
    
    red_cnt = 0
    green_cnt = 0
    history = None
    
    while True:
        
            _, frame = cap.read()
            frame = rotate_img(frame)
            roi = frame.copy()[min_point_list[0][1]:max_point_list[0][1], min_point_list[0][0]:max_point_list[0][0]]
            cv.rectangle(frame, (min_point_list[0][0], min_point_list[0][1]),\
                (max_point_list[0][0], max_point_list[0][1]), (0, 0, 255), 3)

            red_detection_result = red_detection(roi)

            np.where(red_detection_result < 50, 0, red_detection_result)
                
            
            #print("sum :", red_detection_result.sum())


            if red_detection_result.sum() >= 4000: # RED LIGHT
                green_cnt = 0
                if history == "GREEN" or history == None:
                    history = "RED"
                    cv.putText(frame, "DETECTING...", (W//2, H//4), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255),3)
                    cv.imshow("Traffic Light", frame)
                    key = cv.waitKey(30)
                    continue
                
                red_cnt += 1
                if red_cnt >= 15:
                    history = "RED"
                    cv.putText(frame, "RED light", (W//2, H//2), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),3)
            
            else: # GREEN LIGHT
                red_cnt = 0
                if history == "RED" or history == None:
                    history = "GREEN"
                    cv.putText(frame, "DETECTING...", (W//2, H//4), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255),3)
                    cv.imshow("Traffic Light", frame)
                    key = cv.waitKey(30)
                    continue
                
                green_cnt += 1
                if green_cnt >= 15:
                    history = "GREEN"
                    cv.putText(frame, "GREEN light", (W//2, H//2),cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0),3)

            #cv.imshow("HSV Result", red_detection_result)
            cv.imshow("Traffic Light", frame)
            key = cv.waitKey(30)


    cap.release()
    cv.destroyAllWindows()

main()
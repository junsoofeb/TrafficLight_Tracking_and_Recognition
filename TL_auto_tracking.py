import cv2 as cv
import numpy as np
import os
import sys
import natsort
import copy

# 배열 전체 출력을 위해서
np.set_printoptions(threshold=sys.maxsize) 

def show(img):
    cv.imshow("", img)
    cv.waitKey()
    cv.destroyAllWindows()

def color_detect(frame):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)  
        
        lower_black = np.array([0, 0, 0])             
        upper_black = np.array([50, 50, 100])
        #upper_black = np.array([255, 255, 100])
        
        lower_green = np.array([70, 70, 70])             
        upper_green = np.array([90, 255, 255])

        lower_red = np.array([0, 130, 130])             
        upper_red = np.array([10, 255, 255])

        mask_BLACK = cv.inRange(hsv, lower_black, upper_black)
        mask_GREEN = cv.inRange(hsv,lower_green,upper_green)
        mask_RED = cv.inRange(hsv,lower_red,upper_red)

        BLACK_RESULT = cv.bitwise_and(frame, frame, mask = mask_BLACK)
        GREEN_RESULT = cv.bitwise_and(frame, frame, mask = mask_GREEN)
        RED_RESULT = cv.bitwise_and(frame, frame, mask = mask_RED)

        #show(mask_BLACK)
        # show(frame)
        #show(BLACK_RESULT)
        #cv.imshow('GREEN',GREEN_RESULT)
        #cv.imshow('RED',RED_RESULT)
        #cv.waitKey()
        #cv.destroyAllWindows()

        return BLACK_RESULT, GREEN_RESULT, RED_RESULT

def red_detection(frame):
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)  
        
    lower_red = np.array([0, 50, 50])             
    upper_red = np.array([10, 255, 255])
    
    mask_RED = cv.inRange(hsv,lower_red,upper_red)    
    RED_RESULT = cv.bitwise_and(frame, frame, mask = mask_RED)
    
    return RED_RESULT
    
def rotate_img(src):
    height, width, channel = src.shape
    matrix = cv.getRotationMatrix2D((width/2, height/2), -90, 1)
    dst = cv.warpAffine(src, matrix, (width, height))

    return dst

def find_candidate(img):
    candidates = []
    locations = []
    copy = img.copy()
    copy_img = img.copy()
    _, green, red = color_detect(img)

    # 첫 이미지가 빨간불이라면,
    gray = cv.cvtColor(red, cv.COLOR_BGR2GRAY)
    blur = cv.bilateralFilter(gray, 5, 75, 75)
    canny = cv.Canny(blur, 50, 150)
    #show(canny)
    
    kernel = np.ones((10, 10), np.uint8)    
    result = cv.morphologyEx(canny, cv.MORPH_DILATE, kernel)
    #show(result)    
    
    cnts = cv.findContours(result, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]
    for j in range(len(cnts)):
        cnt = cnts[j]
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        box = np.int0(box)        
        
        min_x = min(box[:, 0])
        max_x = max(box[:, 0])
        min_y = min(box[:, 1])
        max_y = max(box[:, 1])
        
        garo = max_x - min_x
        sero = max_y - min_y
        
        ratio = garo / sero
        # 넓이, 가로 길이, 세로길이, 종횡비
    
        # ratio가 1에 가까워야 후보로 설정
        if ratio < 1.2 and ratio > 0.8:
            #cv.drawContours(copy, [box], -1, (255, 0, 0), 4)
            #print("ratio :", ratio)
            mx, my, MX, MY = min_x - 10, min_y - 20, max_x + 10, max_y + 40
            candidate = copy_img[my : MY , mx  : MX ]
            candidates.append(candidate)
            locations.append((mx, my, MX, MY))
        
    #show(copy)

    return candidates, locations

def check_TL(candidates):
    target = None
    target_area = 0
    target_index = 0
    for i, candidate in enumerate(candidates):
        # 검은색 영역 찾고, opening으로 노이즈 제거 후 일정 크기 이상 검은 부분있으면 후보로 설정
        hsv = cv.cvtColor(candidate, cv.COLOR_BGR2HSV)
        lower_hue = np.array([0,0,0])
        upper_hue = np.array([50,50,100])
        mask = cv.inRange(hsv, lower_hue, upper_hue)

        kernel = np.ones((3, 3), np.uint8)    
        result = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        #show(result)    

        cnts = cv.findContours(result, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]
        for j in range(len(cnts)):
            cnt = cnts[j]
            area = cv.contourArea(cnt)

            # area가 100을 넘는 것 중 가장 넓은 것이 후보
            if area > 100 and area > target_area :
                target_area = area
                target = candidate.copy()
                target_index = i
    
    if target is None:
        #print("Black Area ERROR!")
        return None

    return target, target_index
    

def first_frame():
    video_path = '/home/junsoofeb/py_project/traffic_light_detection/traffic_light_1.mp4'
    cap = cv.VideoCapture(video_path)

    _, frame = cap.read()
    frame = rotate_img(frame)
    H, W, _ = frame.shape
    # 신호등 후보와 각 위치정보
    candidates, locations = find_candidate(frame)
    
    # 후보에 대하여 신호등인지 판별
    try:
        target, target_index = check_TL(candidates)
    except: 
        cv.putText(frame, f"Tracking failure...", (100,80), cv.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
        cv.imshow("fail", frame)
        cv.waitKey(0)
        cv.destroyAllWindows()
        sys.exit()

    location = locations[target_index]
    cap.release()
    
    return target, location

def main():
    target, location = first_frame()
    #show(target)
    tracker = cv.TrackerKCF_create()
    video = cv.VideoCapture('/home/junsoofeb/py_project/traffic_light_detection/traffic_light_1.mp4')
    _, frame = video.read()
    frame = rotate_img(frame)
    H, W, _ = frame.shape
    bbox = location
    mx, my, MX, MY = bbox
    '''
    mx, my = bbox[0], bbox[1]
    MX, MY = bbox[0] + bbox[2], bbox[1] + bbox[3]
    '''
    bbox =  mx, my, MX - mx, MY - my
    #print(bbox)
    ok = tracker.init(frame, bbox)
    red_cnt = 0
    green_cnt = 0
    thre = 15
    
    history = None
    last_bbox = None

    failure_cnt = 0
    while True:
        # Read a new frame
        _, frame = video.read()
        frame = rotate_img(frame)

        if failure_cnt > 5:
            failure_cnt = 0
            ok = tracker.init(frame, last_bbox)
            ok, bbox = tracker.update(frame)
            if bbox == (0.0, 0.0, 0.0, 0.0):
                ok = True
                bbox = copy.deepcopy(last_bbox)
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            roi = frame.copy()[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
            cv.rectangle(frame, p1, p2, (0,0,255), 2, 3)
            #print(last_bbox)
            #show(frame)
            #show(roi)
        else:
            # Update tracker
            ok, bbox = tracker.update(frame)
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            roi = frame.copy()[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
            #p1 = (mx, my)
            #p2 = (MX, MY)
            #roi = frame.copy()[my : MY , mx  : MX]
            cv.rectangle(frame, p1, p2, (0,0,255), 2, 3)

        # Draw bounding box
        if ok:
            # Tracking success
            last_bbox = copy.deepcopy(bbox)
            red_detection_result = red_detection(roi)
            np.where(red_detection_result < 50, 0, red_detection_result)      
            #np.where(red_detection_result < 50, 0, red_detection_result)      

            #print(red_detection_result.sum())

            if red_detection_result.sum() >= 3000: # RED LIGHT

                green_cnt -= 15
                if green_cnt < 0:
                    green_cnt = 0
                if history == "GREEN" or history == None:
                    history = "RED"
                    cv.putText(frame, "DETECTING...", (W//2, H//4), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255),3)
                    cv.imshow("Traffic Light", frame)
                    key = cv.waitKey(15)
                    continue
                
                red_cnt += 1
                if red_cnt >= thre:
                    history = "RED"
                    cv.putText(frame, "RED light", (W//2, H//2), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),3)

            else: # GREEN LIGHT
                red_cnt -= 15
                if red_cnt < 0:
                    red_cnt = 0
                if history == "RED" or history == None:
                    history = "GREEN"
                    cv.putText(frame, "DETECTING...", (W//2, H//4), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255),3)
                    cv.imshow("Traffic Light", frame)
                    key = cv.waitKey(15)
                    continue
                
                green_cnt += 1
                if green_cnt >= thre:
                    history = "GREEN"
                    cv.putText(frame, "GREEN light", (W//2, H//2),cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0),3)

            #cv.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        else :
            # Tracking failure
            failure_cnt += 1
            if history == "GREEN":
                cv.putText(frame, "GREEN light", (W//2, H//2),cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0),3)
            else:
                cv.putText(frame, "RED light", (W//2, H//2), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),3)
            cv.putText(frame, f"Tracking failure...", (100,80), cv.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            cv.imshow("Traffic Light", frame)
            key = cv.waitKey(15)
            continue
        #cv.imshow("HSV Result", red_detection_result)
        cv.imshow("Traffic Light", frame)
        key = cv.waitKey(15)


    
main()

import cv2 as cv
import numpy as np
import os
import sys
import natsort

history = None

def show(img):
    cv.imshow("", img)
    cv.waitKey()
    cv.destroyAllWindows()

def rotate_img(src):
    height, width, channel = src.shape
    matrix = cv.getRotationMatrix2D((width/2, height/2), -90, 1)
    dst = cv.warpAffine(src, matrix, (width, height))

    return dst

def red_detection(frame):
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)  
        
    lower_red = np.array([0, 50, 50])             
    upper_red = np.array([10, 255, 255])
    
    mask_RED = cv.inRange(hsv,lower_red,upper_red)    
    RED_RESULT = cv.bitwise_and(frame, frame, mask = mask_RED)
    
    return RED_RESULT
    

#tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

tracker = cv.TrackerKCF_create()

# Read video
video = cv.VideoCapture('/home/junsoofeb/py_project/traffic_light_detection/traffic_light_1.mp4')

# Read first frame.
_, frame = video.read()
frame = rotate_img(frame)
H, W, _ = frame.shape

# Define an initial bounding box
bbox = (405, 139, 445, 208)
# Uncomment the line below to select a different bounding box
bbox = cv.selectROI(frame, False)
cv.destroyAllWindows()
# Initialize tracker with first frame and bounding box
ok = tracker.init(frame, bbox)
red_cnt = 0
green_cnt = 0
thre = 30
while True:
    # Read a new frame
    _, frame = video.read()
    frame = rotate_img(frame)
    # Start timer
    timer = cv.getTickCount()
    # Update tracker
    ok, bbox = tracker.update(frame)
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    roi = frame.copy()[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
    cv.rectangle(frame, p1, p2, (0,0,255), 2, 3)
    

    # Calculate Frames per second (FPS)
    fps = cv.getTickFrequency() / (cv.getTickCount() - timer);
    # Draw bounding box
    if ok:
        # Tracking success
        red_detection_result = red_detection(roi)
        np.where(red_detection_result < 50, 0, red_detection_result)      
        
        if red_detection_result.sum() >= 4000: # RED LIGHT
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
        cv.putText(frame, "Tracking failure detected", (100,80), cv.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
        cv.imshow("Tracking", frame)
        key = cv.waitKey(0)
        cv.destroyAllWindows()
        sys.exit()

    cv.imshow("HSV Result", red_detection_result)
    cv.imshow("Traffic Light", frame)
    key = cv.waitKey(15)
    # Display FPS on frame
    #cv.putText(frame, "FPS : " + str(int(fps)), (100,50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
    # Display result
    #cv.imshow("Tracking", frame)
    # Exit if ESC pressed
    #k = cv.waitKey(1) & 0xff
    #if k == 27 : break

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time
import math
import numpy as np
import pyautogui

screen_w ,screen_h = pyautogui.size()
pyautogui.FAILSAFE=False
canvas = None
latest_result= None

def print_result(result: vision.HandLandmarkerResult,output_image :mp.Image,timestamp_ms:int ):
    global latest_result
    latest_result = result

model_path = 'hand_landmarker.task'

base_options = python.BaseOptions(model_asset_path=model_path)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode = vision.RunningMode.LIVE_STREAM,
    result_callback=print_result,
    min_hand_detection_confidence=.25,
    min_tracking_confidence=.25
)

sm = 0.6
margin=0.2
cl_x,cl_y =0,0

connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (9, 10), (10, 11), (11, 12),         # Middle
    (13, 14), (14, 15), (15, 16),        # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (5, 9), (9, 13), (13, 17)            # Palm base
]

finger_tips  = [4,8,12,16,20]
detector = vision.HandLandmarker.create_from_options(options)

x_prev , y_pres = None,None

cap = cv2.VideoCapture(0)

history={}

while cap.isOpened():
    suc , frame = cap.read()
    if canvas is None:
        canvas = np.zeros_like(frame)
    if not suc:
        break
    
    frame_timestamp_ms = int(time.time()*1000)
    rgb_frame  =cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb_frame)
    detector.detect_async(mp_image,frame_timestamp_ms)
    
    if latest_result is not None and latest_result.hand_landmarks:
        active_ind = []
        for hand_indx,hl in enumerate(latest_result.hand_landmarks):
            active_ind.append(hand_indx)
            
            h,w,_=frame.shape
            thumb = hl[4]
            middle = hl[12]
            
            min_val = margin
            max_value = 1.0-margin
            
            it_x = np.interp(1.0-hl[8].x,[min_val,max_value],[0,screen_w])
            it_y = np.interp(hl[8].y,[min_val,max_value],[0,screen_h])
            
            new_pos_x = (sm*it_x+((1-sm)*cl_x))
            new_pos_y = (sm*it_y+((1-sm)*cl_y))
            cl_x,cl_y= new_pos_x,new_pos_y
            pyautogui.moveTo(int(cl_x) ,int(cl_y),_pause=False)
                      
            t_x ,t_y = int(thumb.x*w) , int(thumb.y*h)
            i_x ,i_y = int(middle.x*w) , int(middle.y*h)
            
    
            distance = math.hypot(i_x-t_x,i_y-t_y)
            
            if distance < 25:
                
                mid_x = ((i_x+t_x)//2) # also current _x 
                mid_y = ((i_y+t_y)//2)
                
                if hand_indx in history:
                    x_prev,y_pres=history[hand_indx]
                    coolors = [[0,255,255],[255,255,0]]
                    color = coolors[hand_indx % len(coolors)]
                    #cv2.line(canvas,(x_prev,y_pres),(mid_x,mid_y),color,5)
                    pyautogui.click()
                
                history[hand_indx] = (mid_x,mid_y)
                

                cv2.circle(frame,(mid_x,mid_y),5,[155,255,0],-1)
                cv2.putText(frame,'PINCH',(i_x,i_y-30),cv2.FONT_HERSHEY_PLAIN,1,(255,255,0),2)  
            else:
                if hand_indx in history:
                    del history[hand_indx]
         
                
            for pair in connections:
                    p1= hl[pair[0]]
                    p2= hl[pair[1]]
                    
                    start_p = (int(p1.x*w),int(p1.y*h))
                    end_p = (int(p2.x*w),int(p2.y*h))
                    
                    cv2.line(frame,start_p,end_p,[0,0,0],1)
                    
            for idx , l, in enumerate(hl):
                
                if idx in finger_tips:
                    color = [0,0,255]
                else: color = [0,255,0]
                
                x = int(l.x*w)
                y = int(l.y*h)
                
                cv2.circle(frame,(x,y),2,color,-1)
                
                cv2.putText(frame,str(idx),(x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,[0,0,0],1)
        for idx in list(history.keys()):
            if idx not in active_ind:
                del history[idx]

    
    frame = cv2.addWeighted(frame,1,canvas,0.5,0)
    cv2.imshow('Detection',frame)
    
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    
    if 0xFF == ord('c'):
        canvas = np.zeros_like(frame)
    
    
detector.close()
cap.release()
cv2.destroyAllWindows()
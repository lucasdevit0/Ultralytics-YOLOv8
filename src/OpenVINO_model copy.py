from ultralytics import YOLO
import cv2 as cv
import numpy as np
import os
import pickle as pkl
import time

from helper import get_bbox_info, bbox_centroid, bbox_id_label, bbox_label, bbox_rectangle

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

cap = cv.VideoCapture('./input/cars.mp4')

# Recording output 
record = False
file_name = 'cars_out'

model = YOLO('./models/yolov8n_openvino_model/')

# Confidence threshold
conf_threshold = 0.3 

# Recording output
if record:
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    size = (frame_width, frame_height)
    cap_out = cv.VideoWriter('output/{}.mp4'.format(file_name), cv.VideoWriter_fourcc('m','p','4','v'),cap.get(cv.CAP_PROP_FPS), size)

frame_counter = 0
# pre_process = []
# inference = []
# post_process = []
frames = []

# # Model speed
# def get_speeds(results, pre_process=pre_process, inference=inference, post_process=post_process):
#     pre_process.append(results.speed['pre_process'])
#     inference.append(results.speed['inference'])
#     post_process.append(results.speed['post_process'])
#     return pre_process, inference, post_process


####################### MAIN LOOP ##############################

while True:
    
    success, frame = cap.read()
    
    if success:

        results = model(frame, stream = True)
        
        #pre_process, inference, post_process = get_speeds(results)
        
        for result in results:

            for box in result.boxes.data.tolist():
                
                # Get bbox info
                x1, y1, x2, y2, class_id, score = get_bbox_info(box)

                # Detection conditions
                if class_id == 2 and score >= conf_threshold:
                    
                    # Class id = 2 -> car
                    bbox_rectangle(x1,y1,x2,y2,frame)
                    bbox_label(x1,y1,x2,y2,frame)
                    bbox_id_label(x1,y1,x2,y2,frame, class_id)
                    
                    # Centroid
                    x_centroid, y_centroid = bbox_centroid(x1, x2, y1, y2,frame)
                
                           
    if record:
        cap_out.write(frame)
        print('Recording -> Frame: {}'.format(frame_counter))
        frame_counter += 1
        frames.append(frame_counter)
    else:
        cv.imshow('img',frame)
        print('Showing -> Frame: {}'.format(frame_counter))
        frame_counter += 1
        frames.append(frame_counter)
        if cv.waitKey(25) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            break
        
####################### MAIN LOOP ##############################



cap.release()
cv.destroyAllWindows()

if record:
    cap_out.release()

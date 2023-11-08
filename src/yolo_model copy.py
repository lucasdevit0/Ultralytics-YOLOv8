from ultralytics import YOLO
import cv2 as cv
import numpy as np
import os
import pickle as pkl
import time

#from speed import speed_graph

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

cap = cv.VideoCapture('./input/ferramentaria.mp4')

# Recording output 
record = False
file_name = 'cars_out'

model = YOLO('./models/yolov8n_openvino_model/')

# Confidence threshold
conf_threshold = 0.3 

# Get bbox information
def get_bbox_info(box):
    '''Input: box (bbox coming from results.boxes.data.tolist())
    Output: 
    x1, y1, x2, y2 -> bbox coordinates
    class_id -> class from coco_classes.txt
    score -> confidence score of bbox detection'''
    x1, y1, x2, y2, score, class_id = box
    x1 = int(x1)
    x2 = int(x2)
    y1 = int(y1)
    y2 = int(y2)
    class_id = int(class_id)
    return x1, y1, x2, y2, class_id, score

# Find bbox centroid
def bbox_centroid(x1,x2,y1,y2,frame):
    #find center of mass
    x_point = int(x1 + ((x2-x1)/2))
    y_point = int(y1 + ((y2-y1)/2))

    #draw centroid
    cv.circle(img = frame,
              center = (x_point,y_point),
              radius = 2,
              color = (255,255,255), 
              thickness = 2, 
              lineType = cv.LINE_8
    )
    return x_point, y_point

    
# Recording output
if record:
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    size = (frame_width, frame_height)
    cap_out = cv.VideoWriter('output/{}.mp4'.format(file_name), cv.VideoWriter_fourcc('m','p','4','v'),cap.get(cv.CAP_PROP_FPS), size)

frame_counter = 0
pre_process = []
inference = []
post_process = []
frames = []

# Model speed
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

                    # object bbox rectangle
                    cv.rectangle(img = frame,
                                pt1 = (int(x1), int(y1)),
                                pt2 = (int(x2), int(y2)),
                                color = (0,100,255),
                                thickness = 2,
                                lineType = cv.LINE_8)
                    
                    # bbox label rectangle
                    cv.rectangle(img = frame,
                                pt1 = (int(x1) -10, int(y1)),
                                pt2 = (int(x1)+ 30, int(y1)- 15),
                                color = (255,255,255),
                                thickness = -1,
                                lineType = cv.LINE_8)
                    
                    # bbox label text
                    cv.putText(img = frame,
                            text = 'ID:{}'.format(class_id), 
                            org = (int(x1) - 5, int(y1) - 3),
                            fontFace = cv.FONT_HERSHEY_SIMPLEX,
                            fontScale = 0.4,
                            color = (0,0,0), 
                            thickness = 1, 
                            lineType = cv.LINE_AA)
                    
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

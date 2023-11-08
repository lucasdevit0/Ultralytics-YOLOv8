from ultralytics import YOLO
import cv2 as cv
import os


import helper

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

video_path = './input/cars.mp4'
cap = cv.VideoCapture(video_path)

model = YOLO('./models/yolov8n.pt')

# Confidence threshold
conf_threshold = 0.3 

frame_counter = 0
frames = []

####################### MAIN LOOP ##############################

while cap.isOpened():
    
    success, frame = cap.read()
    
    if success:

        results = model(frame, stream = True)
        
        #pre_process, inference, post_process = get_speeds(results)
        
        for result in results:

            for box in result.boxes.data.tolist():
                
                # Get bbox info
                x1, y1, x2, y2, class_id, score = helper.get_bbox_info(box)

                # Detection conditions
                if class_id == 2 and score >= conf_threshold:
                    
                    # Class id = 2 -> car
                    helper.bbox_rectangle(x1,y1,x2,y2,frame)
                    helper.bbox_class_id_label(x1,y1,x2,y2,frame, class_id)
                    
                    # Centroid
                    x_centroid, y_centroid = helper.bbox_centroid(x1, x2, y1, y2,frame)
                
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



import cv2 as cv

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


def bbox_rectangle(x1,y1,x2,y2,frame):
    cv.rectangle(img = frame,
        pt1 = (int(x1), int(y1)),
        pt2 = (int(x2), int(y2)),
        color = (191,64,191),
        thickness = 3,
        lineType = cv.LINE_8)
    
    
# bbox label rectangle
def bbox_class_id_label(x1,y1,x2,y2,frame, class_id):
    cv.rectangle(img = frame,
        pt1 = (int(x1), int(y1)),
        pt2 = (int(x1)+ 30, int(y1)- 15),
        color = (255,255,255),
        thickness = -1,
        lineType = cv.LINE_8)

    cv.putText(img = frame,
        text = 'ID:{}'.format(class_id), 
        org = (int(x1) + 5, int(y1) - 3),
        fontFace = cv.FONT_HERSHEY_SIMPLEX,
        fontScale = 0.4,
        color = (0,0,0), 
        thickness = 1, 
        lineType = cv.LINE_AA)

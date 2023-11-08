from ultralytics import YOLO

# Load a YOLOv8n PyTorch model
model = YOLO('models/yolov8n.pt')

# Export the model
model.export(format='openvino')  # creates 'yolov8n_openvino_model/'

# Load the exported OpenVINO model
ov_model = YOLO('models/')

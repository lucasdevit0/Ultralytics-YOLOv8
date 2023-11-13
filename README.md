# Ultralytics-YOLOv8 Collection 👁️

Welcome to the YOLOv8 Implementations repository, where we explore various enhancements and optimizations for the YOLOv8 object detection algorithm. This repository includes different implementations that go beyond the standard YOLOv8, incorporating additional features and optimizations to improve performance and functionality.

# 🗒️Contents

| **Scripts** | **Description** |
|:-------------|:----------------|
| [**download_models.py**](https://github.com/lucasdevit0/Ultralytics-YOLOv8/blob/main/src/download_models.py) | This file can be used to download different YOLO models and export it as onnx and openvino optimized models.   |
| [**helper.py**](https://github.com/lucasdevit0/Ultralytics-YOLOv8/blob/main/src/helper.py) | This file can be used to interact with results = model.predict(). You can simply plot bounding boxes, class_id labels and centroids by calling this helper file. |
| [**OpenVINO_model.py**](https://github.com/lucasdevit0/Ultralytics-YOLOv8/blob/main/src/OpenVINO_model.py) | Implementation of YOLOv8 prediction on a video file using the openVINO model (optimized for Intel hardware - runs inference 3x faster) |
 | [**yolo_model_recording.py**](https://github.com/lucasdevit0/Ultralytics-YOLOv8/blob/main/src/yolo_model_recording.py) | This file can be used to run YOLOv8 on a video file and export the results as .mp4 |
 | [**yolo_model.py**](https://github.com/lucasdevit0/Ultralytics-YOLOv8/blob/main/src/yolo_model.py) | Most basic implementation of YOLOv8 model on a video stream |
 | [**tolo_tracker.py**](https://github.com/lucasdevit0/Ultralytics-YOLOv8/blob/main/src/yolo_tracker.py) | Implementation of YOLOv8 tracker on a video stream (BotSort or ByteTrack) |
 

 # 🌲Directory structure


The project follows an organized directory structure, ensuring clarity, modularity, and ease of navigation. Here is a breakdown of the structure:

```
Ultralytics-YOLOv8/
├── input/                              - Input video streams
│   └── cars.mp4
├── LICENSE                             - Open-source MIT License
├── models/                             - YOLO, onnx and openvino models
│   ├── yolov8n.onnx
│   ├── yolov8n.pt
│   └── yolov8n_openvino_model/
│       ├── metadata.yaml
│       ├── yolov8n.bin
│       └── yolov8n.xml
├── output/                             - outputs from yolo_model_recording.py
│   └── cars_out.mp4
├── README.md                           - Brief repository description
├── requirements.txt                    - Main dependencies
├── src/                                - Main scripts
│   ├── download_models.py
│   ├── helper.py
│   ├── OpenVINO_model.py
│   ├── yolo_model.py
│   ├── yolo_model_recording.py
│   └── yolo_tracker.py
├── trackers/                           - Tracker files
│   ├── botsort.yaml
│   └── bytetrack.yaml
└── txt/                                - Object Detection class reference
    └── coco_classes.txt
```

# 💻Installation

To get started, you'll need to clone this repository and set up the environment:

```shell
git clone https://github.com/lucasdevit0/Ultralytics-YOLOv8.git
cd Ultralytics-YOLOv8
pip install requirements.txt
```

# 🙌🏼Collaboration

Contributions are welcome! If you have improvements, additional features, or optimizations to share, please submit a pull request. Let's collaborate and make YOLOv8 even more powerful and versatile. Cheers!
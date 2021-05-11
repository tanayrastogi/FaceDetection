# faceDetection
OpenCV module to detect faces in an image using Caffe trained model. 
It uses the cv.dnn module to load the trained models. 
The result is a list of faces on the image with confidence and bounding box. 

# Models
Currently tested on,
    - SSD RESNET [res10_300x300_ssd_iter_140000]

### Usage
The module needs following files to run in the folder *models/*
-   Proto File      (.prototxt)
-   Caffee Model    (.caffemodel)

### REFERENCE
- [Face detection with OpenCV and deep learning - PyImageSearch](https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/)
- [OpenCV Face Detection](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector)
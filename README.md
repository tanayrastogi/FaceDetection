# FaceDetection
OpenCV module to detect faces in an image using pre-trained model. 
The result is a list of faces on the image with confidence and bounding box. 

# Models
Currently tested on,
- SSD RESNET [res10_300x300_ssd_iter_140000] trained on Caffe network.
- Multi-Task Cascaded Convolutional Neural Network (MTCNN) model with Tenorflow backend.
- RetinaNet with Tensorflow backend.

### Usage
The module needs following files to run in the folder *models/* to run the Caffee net.
-   Proto File      (.prototxt)
-   Caffe Model    (.caffemodel)

For MTCNN, the package can be installed using [Pip](https://pypi.org/project/mtcnn/).
This also need to have Tensorflow installed. 

For RetinaFace, the package can be installed using [Pip](https://pypi.org/project/retina-face/).
This also need to have Tensorflow installed. 

### REFERENCE
- [Face detection with OpenCV and deep learning - PyImageSearch](https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/)
- [OpenCV Face Detection](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector)
- [Machine Learning Mastery](https://machinelearningmastery.com/how-to-perform-face-detection-with-classical-and-deep-learning-methods-in-python-with-keras/)
- [Sefik Ilkin Serengil Vlog](https://sefiks.com/2021/04/27/deep-face-detection-with-retinaface-in-python/)

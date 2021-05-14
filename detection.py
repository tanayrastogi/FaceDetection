###################################
# Face Detection using 
# RESNET SSD Model 
# trained using Caffe DL Framework
####################################
class CaffeModel:
    import cv2  
    # Python packages
    def __init__(self, modelpath, protopath, confidence=0.5):
        """
        INPUT
            modelpath(str):     Path to the model file.
            protopath(str):     Path to the proto file.
            confidence(float):  Base confidence level to detect faces from the model.
        """ 
        # Variables
        self.BASE_CONFIDENCE = confidence
        
        # Load model
        self.__load_model(modelpath, protopath)

    ## Utility functions ##
    def __check_filepath(self, filepath):
        import os
        if not os.path.isfile(filepath):
           raise Exception("The {} does not exit!".format(os.path.basename(filepath)))

    ## Load Model ##
    def __load_model(self, modelpath, protopath):
        # Sanity checks
        self.__check_filepath(modelpath)
        self.__check_filepath(protopath)

        # # Loading the model
        print("[FacD] Setting up the Caffe RESNet SSD model for face detection ...", end=" ")
        self.MODEL = self.cv2.dnn.readNetFromCaffe(protopath, modelpath)
        print("Done!")

    ## Detections ##
    def detect(self, image, imgName=None):
        """
        INPUT:
            image(numpy.ndarray)    :Numpy image array with 3-channels 
            imgName(str)            :Name of the image. Note: Only for printing purposes.
        """
        import numpy as np
        
        # Return 
        faceDetection = list()
        
        if imgName is not None:
            print("\n[FacD] Detecting objects in " + str(imgName) + "...")
        else:
            print("\n[FacD] Detecting objects ...")

        # Height and width of the image
        (height, width)  = image.shape[:2]

        # Construct an input blob for the image
        # by resizing to a fixed 300x300 pixels and then normalizing it
        print("Creating blob, ", end=" ")
        blob = self.cv2.dnn.blobFromImage(image=self.cv2.resize(image, (300, 300)),
                                     scalefactor=1.0,
                                     size=(300, 300),
                                     mean=(104.0, 177.0, 123.0))

        print("Getting predictions", end=" ")
        self.MODEL.setInput(blob)
        detections = self.MODEL.forward()
        print("Done!")
        print("[FacD] Detections shape: {}".format(detections.shape))

        # After getting all the detections, gather label and bounding box.
        # based on base_confidence level
        detections = detections[0,0]
        for detection in detections:
            # Get confidence
            confidence = float(detection[2])

            # If confidence level is greater than the base, then extract bbox for the face
            if confidence>self.BASE_CONFIDENCE:

                # Box dimensions for detection
                box = detection[3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")

                ret_dict = {"label": "face",
                            "bbox":(startX, startY, endX, endY),
                            "confidence":confidence}

                # Add to the returned list
                faceDetection.append(ret_dict)
        return faceDetection

###################################
# Face Detection using 
# Multi-Task Cascaded Convolutional Neural Network (MTCNN) 
# using Tenserflow as backend
####################################
class MTCNN:
    import cv2
    def __init__(self, confidence=0.5):
        """
        Reference:
        https://machinelearningmastery.com/how-to-perform-face-detection-with-classical-and-deep-learning-methods-in-python-with-keras/

        INPUT
            confidence(float):  Base confidence level to detect faces from the model.
        """ 
        # Variables
        self.BASE_CONFIDENCE = confidence
        
        # Load model
        self.__load_model()

    ## Load Model
    def __load_model(self):
        # # Loading the model
        print("[FacD] Setting up the Tensorflow MTCNN model for face detection ...", end=" ")
        from mtcnn.mtcnn import MTCNN
        self.MODEL = MTCNN()
        print("Done!")

    ## Detections ##
    def detect(self, image, imgName=None, imgShape=None):
        """
        INPUT:
            image(numpy.ndarray)    :Numpy image array with 3-channels 
            imgName(str)            :Name of the image. Note: Only for printing purposes.
            imgShape(tuple)         :Start and end coordinate of the image as (startX, startY, endX, endY)
        """
        # Return 
        faceDetection = list()
        
        if imgName is not None:
            print("\n[FacD] Detecting objects in " + str(imgName) + "...")
        else:
            print("\n[FacD] Detecting objects ...")

        # detect faces in the image
        detections = self.MODEL.detect_faces(self.cv2.cvtColor(image, self.cv2.COLOR_BGR2RGB))
        for detection in detections:
            # If confidence level is greater than the base, then extract bbox for the face
            confidence = detection["confidence"]
            
            if confidence>self.BASE_CONFIDENCE:    
                x, y, width, height = detection["box"]
                if imgShape is not None:
                    startX = x + imgShape[0]
                    startY = y + imgShape[1]
                    endX   = startX + width
                    endY   = startY + height
                else:
                    startX = x
                    startY = y
                    endX   = startX + width
                    endY   = startY + height
                faceDetection.append({"label": "face",
                                      "confidence": detection["confidence"],
                                      "bbox": (startX, startY, endX, endY)})
        return faceDetection

###################################
# Face Detection using 
# RetinaFace designed by Insightface project
# using Tenserflow as backend
####################################
class RetinaFace:
    import cv2
    def __init__(self, confidence=0.5):
        """
        Reference:
        https://pypi.org/project/retina-face/

        INPUT
            confidence(float):  Base confidence level to detect faces from the model.
        """ 
        # Variables
        self.BASE_CONFIDENCE = confidence
        
        # Load model
        self.__load_model()

    ## Load Model ##
    def __load_model(self):
        # # Loading the model
        print("[FacD] Setting up the Tensorflow RetinaFace model for face detection ...", end=" ")
        from retinaface import RetinaFace
        self.MODEL = RetinaFace
        print("Done!")


    ## Detections ##
    def detect(self, image, imgName=None, imgShape=None):
        """
        INPUT:
            image(numpy.ndarray)    :Numpy image array with 3-channels 
            imgName(str)            :Name of the image. Note: Only for printing purposes.
            imgShape(tuple)         :Start and end coordinate of the image as (startX, startY, endX, endY)
        """
        # Return 
        faceDetection = list()
        
        if imgName is not None:
            print("\n[FacD] Detecting objects in " + str(imgName) + "...")
        else:
            print("\n[FacD] Detecting objects ...")

        # detect faces in the image
        detections = self.MODEL.detect_faces(image)
        detections = [v for v in detections.values()]
        for detection in detections:
            # If confidence level is greater than the base, then extract bbox for the face
            confidence = detection["score"]
            
            if confidence>self.BASE_CONFIDENCE:
                ret_dict = {"label": "face",
                            "bbox":detection["facial_area"],
                            "confidence":detection["score"]}
                faceDetection.append(ret_dict)
                
        #         if imgShape is not None:
        #             startX = x + imgShape[0]
        #             startY = y + imgShape[1]
        #             endX   = startX + width
        #             endY   = startY + height
        #         else:
        #             startX = x
        #             startY = y
        #             endX   = startX + width
        #             endY   = startY + height
        #         faceDetection.append({"label": "face",
        #                               "confidence": detection["confidence"],
        #                               "bbox": (startX, startY, endX, endY)})

        return faceDetection



if __name__ == "__main__":
    import cv2  
    import numpy as np
    import imutils 
    import time 

    #########
    # Image #
    #########
    imagePath = "test_images/cityscape.jpg"
    image = cv2.imread(imagePath)

    ###########################
    # Testing for Caffe Model #
    ###########################
    # # MODEL Parameters #
    # modelpath = "models/res10_300x300_ssd_iter_140000.caffemodel"
    # protopath = "models/deploy_lowres.prototxt.txt"
    # model = CaffeModel(modelpath, protopath, confidence=0.4)

    # # Detections
    # faces = model.detect(image, "Test")

    ###########################
    # Testing for MTCNN Model #
    ###########################
    model = MTCNN(confidence=0.1)

    # Detections
    faces = model.detect(image, "Test")

    ################################
    # Testing for RetinaFace Model #
    ################################
    # model = RetinaFace(confidence=0.1)

    # # Detections
    # faces = model.detect(image, "Test")

    
    ######################
    # To show the result #
    ######################    
    # Plot bounding boxes on image
    for face in faces:
        (startX, startY, endX, endY) = face["bbox"]
        confidence = face["confidence"]
        # Rectangle around the objects detected
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        label = "{:.2f}".format(confidence)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    (height, width, channel) = image.shape
    text = "Number of faces: {}".format(len(faces))
    cv2.putText(image, text, (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # show the output image
    time.sleep(0.1)
    cv2.imshow("Output", imutils.resize(image, width=1280))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
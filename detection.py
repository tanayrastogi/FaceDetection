# Python packages
import numpy as np
import cv2
import os

class FaceDetection:
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
        
    #######################
    ## Utility functions ##
    #######################
    def __check_filepath(self, filepath):
        if not os.path.isfile(filepath):
           raise Exception("The {} does not exit!".format(os.path.basename(filepath)))

    ################
    ## Load Model ##
    ################
    def __load_model(self, modelpath, protopath):
        # Sanity checks
        self.__check_filepath(modelpath)
        self.__check_filepath(protopath)

        # # Loading the model
        print("[FacD] Setting up the Caffe RESNet SSD model for face detection ...", end=" ")
        self.MODEL = cv2.dnn.readNetFromCaffe(protopath, modelpath)
        print("Done!")

    ################
    ## Detections ##
    ################
    def detect(self, image, imgName=None):
        """
        INPUT:
            image(numpy.ndarray)    :Numpy image array with 3-channels 
            imgName(str)            :Name of the image. Note: Only for printing purposes.
        """
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
        blob = cv2.dnn.blobFromImage(image=cv2.resize(image, (300, 300)),
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

                ret_dict = {"bbox":(startX, startY, endX, endY),
                            "confidence":confidence}

                # Add to the returned list
                faceDetection.append(ret_dict)
        return faceDetection

if __name__ == "__main__":
    # MODEL Parameters #
    modelpath = "models/res10_300x300_ssd_iter_140000.caffemodel"
    protopath = "models/deploy_lowres.prototxt.txt"

    # Model
    model = FaceDetection(modelpath, protopath, confidence=0.1)

    # Image
    imagePath = "fb-17fb02b1b414482eacef639472ac395c.jpg"
    image = cv2.imread(imagePath)
    # Detections
    faces = model.detect(image, "Test")

    # Plot bounding boxes on image
    color = np.random.uniform(0, 255, size=(1, 3)).flatten()
    for face in faces:
        (startX, startY, endX, endY) = face["bbox"]
        confidence = face["confidence"]

        # Rectangle around the objects detected
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
        label = "{:.2f}%".format(confidence)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # show the output image
    import time 
    time.sleep(0.1)
    cv2.imshow("Output", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
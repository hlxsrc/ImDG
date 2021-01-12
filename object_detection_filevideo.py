# usage example
# python object_detection_filevideo.py \
#   --object car \
#   --prototxt MobileNetSSD_deploy.prototxt.txt \
#   --model MobileNetSSD_deploy.caffemodel \
#   --video $HOME/Downloads/videos/Sentra.mp4 \
#   --output output/ \
#   --confidence 0.7

# import the necessary packages
from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-obj", "--object", required=True,
                help="class name of the object")
ap.add_argument("-p", "--prototxt", required=True,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
                help="minimum probability to filter weak detections")
ap.add_argument("-v", "--video",
                help="path to the video file")
ap.add_argument("-o", "--output", required=True,
                help="path to output directory")
args = vars(ap.parse_args())

# check for output directory
if not os.path.exists(args["output"]):
    os.makedirs('output')

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# find class id
class_id = CLASSES.index(args["object"])

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# handles video file processing
vs = FileVideoStream(args["video"]).start()

# calculate the number of pictures taken (ROI)
total = 0

# initialize the FPS counter
fps = FPS().start()

# global x, y, w, h (for ROI)
x = y = w = h = 0

# loop over the frames from the video stream
while vs.more():

    # grab the frame from the threaded video stream,
    # clone it (so we can write to disk) and then
    # resize it to have a maximum width of 400 pixels
    frame = vs.read()
    orig = frame.copy()
    frame = imutils.resize(frame, width=400)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):

        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:

            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])

            # condition to only detect cars
            if CLASSES[idx] == CLASSES[class_id]:

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # draw the prediction on the frame
                label = "{}: {:.2f}%".format(CLASSES[idx],
                                             confidence * 100)

                # uncomment if bounding box is necessary
                # cv2.rectangle(frame, (startX, startY), (endX, endY),
                #              COLORS[idx], 2)

                # uncomment if label is necessary
                # y = startY - 15 if startY - 15 > 15 else startY + 15
                # cv2.putText(frame, label, (startX, y),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

                # get dimensions of the roi
                x = startX
                y = startY
                w = endX - startX
                h = endY - startY

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `s` key was pressed, write the *original* frame to disk
    # so we can later process it and use it for car recognition
    if key == ord("s"):
        p = os.path.sep.join([args["output"], "{}.png".format(
            str(total).zfill(5))])

        # extract the ROI of the *reducted* car
        car = imutils.resize(frame[y:y + h, x:x + w], width=256)
        cv2.imwrite(p, car)

        total += 1

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

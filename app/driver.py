try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser

import cv2.cv2 as cv
from imutils.video.videostream import VideoStream

from time import sleep
import argparse
import sys
import random
import numpy as np
import os.path
import matplotlib
from glob import glob
frame_count = 0
frame_count_out = 0
confThreshold = 0.5
nmsThreshold = 0.4
inpWidth = 416
inpHeight = 416

def _input (file):
     config=ConfigParser()
     config.read(file)
     classesFile= str(config.get('path', 'names_path'))
     modelConfiguration = str(config.get('path', 'config_file'))
     modelWeights =  str(config.get('path', 'model_path'))
     obj =  str(config.get('path', 'obj_find'))
     return  classesFile,modelConfiguration,modelWeights,obj 

def getOutputsNames(net):

    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def drawPred(classId, conf, left, top, right, bottom,classes,frame,finding_name):

    global frame_count
    cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 5)
    label = '%.2f' % conf
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    labelSize, baseLine = cv.getTextSize(
        label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])

    label_name, label_conf = label.split(':')
    if label_name == str(finding_name):
        #cv.rectangle(frame, (left, top - round(5*labelSize[1])), (left + round(7*labelSize[0]), top + baseLine), (0, 0, 255), cv.FILLED)
        cv.putText(frame, label, (left, top),
                   cv.FONT_HERSHEY_SIMPLEX, 1.00, (0, 0, 255), 3)
        frame_count += 1
    print (label)
    return frame        


def postprocess(frame, outs,yes,_not,obj,classes):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    global frame_count_out
    frame_count_out = 0
    classIds = []
    confidences = []
    boxes = []
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                # print(classIds)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    count_person = 0
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]

        frame = drawPred(
            classIds[i], confidences[i], left, top, left + width, top + height,classes,frame,str(obj))

        my_class = str(obj)
       
        unknown_class = classes[classId]

        if my_class == unknown_class:
            count_person += 1
    # if(frame_count_out > 0):
    return frame
#vs = VideoStream(src=2).start()



def main():
    classesFile,modelConfiguration,modelWeights,obj= _input('C:\\Users\\Seva\\Desktop\\videoprocessing testing\\app\\ini\\or-helmet_detection.ini')
    classes = None
    with open(classesFile, 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')
    frame_count = 0
    frame_count_out = 0
    confThreshold = 0.5
    nmsThreshold = 0.4
    inpWidth = 416
    inpHeight = 416
    yes=0
    _not=0
    net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    vs = cv.VideoCapture("vid.mp4") 

    while True:
                    ret,frame = vs.read()
    
                    frame_count = 0
                    frame=cv.rotate(frame,cv.ROTATE_90_COUNTERCLOCKWISE)
                    blob = cv.dnn.blobFromImage(
                        frame, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
                    net.setInput(blob)

                    outs = net.forward(getOutputsNames(net))

                    frame=postprocess(frame, outs,yes,_not,obj,classes)

                    t, _ = net.getPerfProfile()
                    cv.imshow("video",frame)
                    if cv.waitKey(1) & 0xFF == ord('q'):
                                break
                    
    vs.release()

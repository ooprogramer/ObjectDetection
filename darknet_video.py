from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
from imutils.video import FPS

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (255, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [255, 255, 0], 1)
    return img


netMain = None
metaMain = None
altNames = None


def YOLO():

    global metaMain, netMain, altNames
    configPath = "./cfg/yolov3.cfg"
    weightPath = "./yolov3.weights"
    metaPath = "./cfg/coco.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" + os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" + os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" + os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass

    fps = FPS().start()

    frame = []
    cap0 = [cv2.VideoCapture("rtsp://210.89.190.107:10565/PSIA/Streaming/channels/0"), '10565']
    cap1 = [cv2.VideoCapture("rtsp://210.89.190.107:10566/PSIA/Streaming/channels/0"), '10566']
    cap2 = [cv2.VideoCapture("rtsp://210.89.190.107:10567/PSIA/Streaming/channels/0"), '10567']
    cap3 = [cv2.VideoCapture("rtsp://210.89.190.107:10568/PSIA/Streaming/channels/0"), '10568']
    cap4 = [cv2.VideoCapture("rtsp://210.89.190.107:10569/PSIA/Streaming/channels/0"), '10569']
    cap5 = [cv2.VideoCapture("rtsp://210.89.190.107:10570/PSIA/Streaming/channels/0"), '10570']
    cap6 = [cv2.VideoCapture("rtsp://210.89.190.107:10571/PSIA/Streaming/channels/0"), '10571']
    cap7 = [cv2.VideoCapture("rtsp://210.89.190.107:10572/PSIA/Streaming/channels/0"), '10572']
    cap8 = [cv2.VideoCapture("rtsp://210.89.190.107:10573/PSIA/Streaming/channels/0"), '10573']
    #cap9 = [cv2.VideoCapture("rtsp://210.89.190.107:10574/PSIA/Streaming/channels/0"), '10574']
    cap10 = [cv2.VideoCapture("rtsp://210.89.190.107:10575/PSIA/Streaming/channels/0"), '10575']
    frame = [cap0, cap1, cap2, cap3, cap4, cap5, cap6, cap7, cap8, cap10]

    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain), darknet.network_height(netMain),3)
    while True:
        prev_time = time.time()
        fps.update()
        fps.stop()
        
        imagelist = []
        for i in frame:
            temp = yolo_image(i[0], darknet_image)
            imagelist.append(temp)

        for k in range(0,10):
            cv2.putText(imagelist[k], "FPS : " + "{:.2f}".format(fps.fps()), (25, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5 ,(255, 255, 255), 1)
            cv2.imshow(frame[k][1], imagelist[k])
        

        print("{:.4f} seconds".format(time.time()-prev_time))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for j in frame:
        j[0].release()

def yolo_image(cap, darknet_image):
    ret, frame_read = cap.read()
    frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb,(darknet.network_width(netMain),darknet.network_height(netMain)),interpolation=cv2.INTER_LINEAR)
    darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())
    detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
    image = cvDrawBoxes(detections, frame_resized)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

if __name__ == "__main__":
    YOLO()

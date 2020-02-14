#import
import cv2
import numpy as np
import os
import time


#yolo setting
BasePath = "../yolo-coco"
BaseConfidence = 0.3  #0.3
Base_threshold = 0.2  #0.3

def YOLOINIT():
	# load the COCO class labels our YOLO model was trained on
	labelsPath = os.path.sep.join([BasePath, "coco.names"])

	global LABELS
	LABELS = open(labelsPath).read().strip().split("\n")

	# derive the paths to the YOLO weights and model configuration
	weightsPath = os.path.sep.join([BasePath, "yolov3.weights"])
	configPath = os.path.sep.join([BasePath, "yolov3.cfg"])

	# load our YOLO object detector trained on COCO dataset (80 classes)
	print("[INFO] loading YOLO from disk...")
	global net
	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

	# determine only the *output* layer names that we need from YOLO
	global ln
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	# initialize the video stream, pointer to output video file, and
	# frame dimensions

	(W, H) = (None, None)
#end YOLOINIT()

def YOLOTINYINIT():
	labelsPath = os.path.sep.join([BasePath, "coco.names"])

	global LABELS
	LABELS = open(labelsPath).read().strip().split("\n")

	weightsPath = os.path.sep.join([BasePath, "yolov3-tiny.weights"])
	configPath = os.path.sep.join([BasePath, "yolov3-tiny.cfg"])

	print("[INFO] loading YOLO_TINY from disk...")
	global net
	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

	global ln
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	(W, H) = (None, None)
#end YOLOINIT()

def YOLO_Detect(frame):
	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	#print("[INFO] {:.6f} seconds".format(end - start))
	return layerOutputs,start,end
#end YOLO_Detect()


def YOLO_BOX_INFO(frame,layerOutputs):

	H, W = frame.shape[:2]  ## 1920 x 1080
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > BaseConfidence:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates,
				# confidences, and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, BaseConfidence, Base_threshold)
	return idxs, boxes , classIDs, confidences

def Position(idxs,classIDs,boxes,Vehicle_x, Vehicle_y, Vehicle_w, Vehicle_h):

    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            if classIDs[i] == 2 or classIDs[i] == 7:##검출된 이미지가 차량(2) 또는 트럭(7) 인경우에
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # 검출된 번호에 맞게 차량의 위치, 크기 정보 대입
                Vehicle_x.append(x); Vehicle_y.append(y); Vehicle_w.append(w); Vehicle_h.append(h)

    return Vehicle_x, Vehicle_y, Vehicle_w, Vehicle_h




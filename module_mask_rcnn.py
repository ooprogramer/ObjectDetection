# import the necessary packages
import numpy as np
import time
import cv2
import os

RcnnPath = "mask-rcnn-coco"
RcnnConfidence = 0.6
Rcnn_threshold = 0.3

visualize = 0

COLOR = [0,200,200]

def init():
	labelsPath = os.path.sep.join([RcnnPath, "object_detection_classes_coco.txt"])
	global LABELS_RCNN
	LABELS_RCNN = open(labelsPath).read().strip().split("\n")

	weightsPath = os.path.sep.join([RcnnPath, "frozen_inference_graph.pb"])
	configPath = os.path.sep.join([RcnnPath, "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])

	print("[INFO] loading Mask R-CNN from disk...")
	global net_RCNN
	net_RCNN = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)


def detect(image):
	blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
	net_RCNN.setInput(blob)
	start = time.time()
	(boxes, masks) = net_RCNN.forward(["detection_out_final", "detection_masks"])
	end = time.time()

	car_num = 0
	for i in range(0, boxes.shape[2]):
		classID = int(boxes[0, 0, i, 1])
		confidence = boxes[0, 0, i, 2]
		if (classID==2 or classID==7) and (confidence>RcnnConfidence):
			car_num += 1
			#output(image,boxes,masks)

	print("[INFO] number of car: {}".format(car_num))
	#print("[INFO] boxes shape: {}".format(boxes.shape))
	#print("[INFO] masks shape: {}".format(masks.shape))
	#print("[INFO] Mask R-CNN took {:.6f} seconds".format(end - start))
	#print("[INFO] number of object: {}".format(len(boxes[0][0])))

	return boxes, masks, car_num

def output(image, boxes, masks):
	(H, W) = image.shape[:2]
	for i in range(0, boxes.shape[2]):
		classID = int(boxes[0, 0, i, 1])
		confidence = boxes[0, 0, i, 2]

		if confidence > RcnnConfidence:
			box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
			(startX, startY, endX, endY) = box.astype("int")
			boxW = endX - startX
			boxH = endY - startY
	
			mask = masks[i, classID]
			mask = cv2.resize(mask, (boxW, boxH),
				interpolation=cv2.INTER_NEAREST)
			mask = (mask > Rcnn_threshold)
	
			roi = image[startY:endY, startX:endX]
			
			extract(roi,mask)

			roi = roi[mask]
			color = np.array(COLOR).astype("uint8")
			blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")
	
			image[startY:endY, startX:endX][mask] = blended
			color = [int(c) for c in color]
			cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
	
			text = "{}: {:.4f}".format(LABELS_RCNN[classID], confidence)
			cv2.putText(image, text, (startX, startY - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)
	
	
	#image = cv2.resize(image, (1280,720), interpolation=cv2.INTER_AREA) #1920,1080 -> 1280,720 -> 960,540
	cv2.imshow("Output", image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	

def extract(roi, mask):
	if visualize > 0:
		visMask = (mask * 255).astype("uint8")
		instance = cv2.bitwise_and(roi, roi, mask=visMask)
	
		cv2.imshow("ROI", roi)
		cv2.imshow("Mask", visMask)
		cv2.imshow("Segmented", instance)
		cv2.waitKey(0)



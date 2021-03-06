# import the necessary packages
import numpy as np
import time
import cv2
import os

BasePath = "mask-rcnn-coco"
BaseConfidence = 0.8
Base_threshold = 0.3

COLOR = [0,200,200]

def init():
	labelsPath = os.path.sep.join([BasePath, "object_detection_classes_coco.txt"])
	global LABELS
	LABELS = open(labelsPath).read().strip().split("\n")

	weightsPath = os.path.sep.join([BasePath, "frozen_inference_graph.pb"])
	configPath = os.path.sep.join([BasePath, "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])

	print("[INFO] loading Mask R-CNN from disk...")
	global net
	net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)


def detect(image):
	blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	(boxes, masks) = net.forward(["detection_out_final", "detection_masks"])
	end = time.time()

	car_num = 0
	for i in range(0, boxes.shape[2]):
		classID = int(boxes[0, 0, i, 1])
		confidence = boxes[0, 0, i, 2]
		if (classID==2 or classID==7) and (confidence>BaseConfidence):
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

		if confidence > BaseConfidence:
			box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
			(startX, startY, endX, endY) = box.astype("int")
			boxW = endX - startX
			boxH = endY - startY
	
			mask = masks[i, classID]
			mask = cv2.resize(mask, (boxW, boxH),
				interpolation=cv2.INTER_NEAREST)
			mask = (mask > Base_threshold)
	
			roi = image[startY:endY, startX:endX]
			
			#extract(roi,mask)

			roi = roi[mask]
			color = np.array(COLOR).astype("uint8")
			blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")
	
			image[startY:endY, startX:endX][mask] = blended
			color = [int(c) for c in color]
			cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
	
			text = "{}: {:.4f}".format(LABELS[classID], confidence)
			cv2.putText(image, text, (startX, startY - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)

	cv2.imshow("Output", image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	

def extract(roi, mask):
	visMask = (mask * 255).astype("uint8")
	instance = cv2.bitwise_and(roi, roi, mask=visMask)
	
	cv2.imshow("ROI", roi)
	cv2.imshow("Mask", visMask)
	cv2.imshow("Segmented", instance)
	cv2.waitKey(0)



# import the necessary packages
import numpy as np
import random
import time
import cv2
import os

input_image = "./image/19_park.png"
BasePath = "/home/mtov/git/mask-rcnn-coco"
BaseConfidence = 0.5
Base_threshold = 0.3

visualize = 0

COLOR = [200,200,200]
image = cv2.imread(input_image)


def main():
	init()
	a = image[150:330, 280:750]
	b = image[300:800, 0:600]
	c = image[170:350, 1180:1500]
	d = image[350:800, 1280:1920]
	boxes, masks, car_num = detect(d)
	output(d,boxes,masks)

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


	#print("[INFO] Mask R-CNN took {:.6f} seconds".format(end - start))
	print("[INFO] number of object: {}".format(len(boxes[0][0])))
	print("[INFO] number of car: {}".format(car_num))
	#print("[INFO] boxes shape: {}".format(boxes.shape))
	#rint("[INFO] masks shape: {}".format(masks.shape))

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
			
			extract(roi,mask)

			roi = roi[mask]
			color = np.array(COLOR).astype("uint8")
			blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")
	
			image[startY:endY, startX:endX][mask] = blended
			color = [int(c) for c in color]
			cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
	
			text = "{}: {:.4f}".format(LABELS[classID], confidence)
			cv2.putText(image, text, (startX, startY - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
	
	
	#image = cv2.resize(image, (1280,720), interpolation=cv2.INTER_AREA) #1920,1080 -> 1280,720 -> 960,540
	cv2.imshow("Output", image)
	cv2.waitKey(0)

def extract(roi, mask):
	if visualize > 0:
		visMask = (mask * 255).astype("uint8")
		instance = cv2.bitwise_and(roi, roi, mask=visMask)
	
		cv2.imshow("ROI", roi)
		cv2.imshow("Mask", visMask)
		cv2.imshow("Segmented", instance)
		cv2.waitKey(0)

if __name__ == '__main__':
    main()


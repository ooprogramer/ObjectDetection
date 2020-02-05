#import
import cv2
import numpy as np
from colors import *
import imutils
from imutils.video import FPS
import time
import os
import math
import matplotlib.pyplot as plt

#global
Input_Video = "../video/22-2.mp4"

#background substraction 을 위한 이미지.
first_frame = cv2.imread("image/22.png")
first_frame = cv2.resize(first_frame, (1920, 1080), interpolation=cv2.INTER_CUBIC)

first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
first_gray = cv2.GaussianBlur(first_gray, (5, 5), 0)

BasePath = "../yolo-coco"
BaseConfidence = 0.3  #0.3
Base_threshold = 0.2  #0.3

def main():
    fps = FPS().start()

    cap = cv2.VideoCapture(Input_Video)

    YOLOINIT()

    f_num = 0

    RED_cnt_1 = 0; BLUE_cnt_1 = 0
    initBB_1 = None; tracker_1 = None

    RED_cnt_2 = 0; BLUE_cnt_2 = 0
    initBB_2 = None; tracker_2 = None

    RED_cnt_3 = 0; BLUE_cnt_3 = 0
    initBB_3 = None; tracker_3 = None

    RED_cnt_4 = 0; BLUE_cnt_4 = 0
    initBB_4 = None; tracker_4 = None

    while(cap.isOpened()):
        f_num =f_num +1
        #print("F : ", f_num)

        (grabbed, frame) = cap.read()

        if f_num % 2 == 0:
            blank_image = np.zeros((64, 1920, 3), np.uint8)
            frame[0:64, 0:1920] = blank_image

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

            difference = cv2.absdiff(first_gray, gray_frame)
            _, difference = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)

            mask3 = cv2.cvtColor(difference, cv2.COLOR_GRAY2BGR)  # 3 channel mask
            Substracted = cv2.bitwise_and(frame, mask3)

            layerOutputs, start, end = YOLO_Detect(frame)

            idxs, boxes, classIDs, confidences = YOLO_BOX_INFO(frame, layerOutputs, BaseConfidence, Base_threshold)

            Vehicle_x = []; Vehicle_y = []; Vehicle_w = []; Vehicle_h = []

            #차량 포인트 가져옴
            Vehicle_x, Vehicle_y, Vehicle_w, Vehicle_h = Position(idxs, classIDs, boxes, Vehicle_x, Vehicle_y, Vehicle_w, Vehicle_h)

            #차량 포인트 그리기
            Draw_Points(frame, Vehicle_x, Vehicle_y, Vehicle_w, Vehicle_h)

#left-up counter ----------------------------------------------------------------------------------------------------------------------------------------------
            vertices1 = [[[650, 300], [750, 300], [810, 210], [750, 210]]]

            tracker_1, initBB_1, RED_cnt_1, BLUE_cnt_1 = Passing_Counter_Zone(Vehicle_x, Vehicle_y, Vehicle_w, Vehicle_h, initBB_1, frame, tracker_1, Substracted,\
                                      RED_cnt_1, BLUE_cnt_1, vertices1)

            draw_line(frame, vertices1, RED_cnt_1, BLUE_cnt_1)

            pts_1 = detecting_zone(vertices1)
            cv2.polylines(frame, [pts_1], True, (0, 255, 0), 2)

#left-down counter ----------------------------------------------------------------------------------------------------------------------------------------------
            vertices2 = [[[250, 780], [480, 800], [720, 340], [620, 340]]]

            tracker_2, initBB_2, RED_cnt_2, BLUE_cnt_2 = Passing_Counter_Zone(Vehicle_x, Vehicle_y, Vehicle_w, Vehicle_h, initBB_2, frame, tracker_2, Substracted,\
                                      RED_cnt_2, BLUE_cnt_2, vertices2)

            draw_line(frame, vertices2, RED_cnt_2, BLUE_cnt_2)

            pts_2 = detecting_zone(vertices2)
            cv2.polylines(frame, [pts_2], True, (0, 255, 0), 2)

#right-up counter ----------------------------------------------------------------------------------------------------------------------------------------------
            vertices3 = [[[1140, 230], [1080, 230], [1150, 330], [1240, 330]]]

            tracker_3, initBB_3, RED_cnt_3, BLUE_cnt_3 = Passing_Counter_Zone(Vehicle_x, Vehicle_y, Vehicle_w, Vehicle_h, initBB_3, frame, tracker_3, Substracted,\
                                      RED_cnt_3, BLUE_cnt_3, vertices3)

            draw_line(frame, vertices3, RED_cnt_3, BLUE_cnt_3)

            pts_3 = detecting_zone(vertices3)
            cv2.polylines(frame, [pts_3], True, (0, 255, 0), 2)

#right-down counter ----------------------------------------------------------------------------------------------------------------------------------------------
            vertices4 = [[[1270, 360], [1170, 360], [1450, 800], [1660, 800]]]

            tracker_4, initBB_4, RED_cnt_4, BLUE_cnt_4 = Passing_Counter_Zone(Vehicle_x, Vehicle_y, Vehicle_w, Vehicle_h, initBB_4, frame, tracker_4, Substracted,\
                                      RED_cnt_4, BLUE_cnt_4, vertices4)

            draw_line(frame, vertices4, RED_cnt_4, BLUE_cnt_4)

            pts_4 = detecting_zone(vertices4)
            cv2.polylines(frame, [pts_4], True, (0, 255, 0), 2)

            frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_CUBIC) #1920, 1080 -> 1280,720
            #Substracted = cv2.resize(Substracted , (1280, 720), interpolation=cv2.INTER_CUBIC)

            fps.update()
            fps.stop()
            cv2.putText(frame, "FPS : " + "{:.2f}".format(fps.fps()), (25, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255), 2)

            cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    return



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



def YOLO_Detect(frame):
	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	print("[INFO] {:.6f} seconds".format(end - start))
	return layerOutputs,start,end
#end YOLO_Detect()


def YOLO_BOX_INFO(frame,layerOutputs,BaseConfidence,Base_threshold):

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



def Draw_Points(frame,Vehicle_x,Vehicle_y,Vehicle_w,Vehicle_h):
    if len(Vehicle_x) > 0:
        for i in range(0, len(Vehicle_x), 1):
            cv2.circle(frame, (Vehicle_x[i] + int(Vehicle_w[i] / 2), Vehicle_y[i] + Vehicle_h[i]), 5, (0, 255, 0), -1)
            cv2.rectangle(frame, (Vehicle_x[i], Vehicle_y[i]), (Vehicle_x[i]+Vehicle_w[i], Vehicle_y[i]+Vehicle_h[i]), (255, 255, 0), 2)
#end func


def Passing_Counter_Zone(Vehicle_x,Vehicle_y,Vehicle_w,Vehicle_h,initBB,frame,tracker,Substracted,RED_cnt,BLUE_cnt,vertices):
    # 1번 카메라에 대해서만 적용

    # Detecting Zone
    pts = detecting_zone(vertices)

    # 차량 검출시
    for d_num in range(0, len(Vehicle_x)):
        # P 좌표는 프레임에서 디텍팅 포인트 영역
        p_x = Vehicle_x[d_num] + int(Vehicle_w[d_num] / 2)
        p_y = Vehicle_y[d_num] + int(Vehicle_h[d_num])

        crosses = 0  # 교점의 개수(짝수개이면 영역 밖에 존재, 홀수개이면 영역 안에 존재)
        for p in range(0, 4):  # 항상 사각형 이므로 4
            next_p = (p + 1) % 4
            if (pts[p][1] > p_y) != (pts[next_p][1] > p_y):  ##디텍티드 포인트의 Y좌표가 사각형의 두점사이에 존재하면

                # atx가 오른쪽 반직선과의 교점이 맞으면 교점의 개수를 증가시킨다,
                atX = int((pts[next_p][0] - pts[p][0]) * (p_y - pts[p][1]) / (
                            pts[next_p][1] - pts[p][1]) + pts[p][0])
                if p_x < atX:
                    crosses = crosses + 1
                    ##텍스트로 인 아웃 여부
                    # cv2.putText(frame, str(crosses), (atX, p_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,COLOR_GREEN, 3)

        if crosses % 2 == 0:  # 영역 밖에 존재하는 경우
            pass
        elif crosses % 2 == 1:  # 영역 안에 존재하는 경우
            if initBB is None:
                initBB = (Vehicle_x[d_num], Vehicle_y[d_num], Vehicle_w[d_num], Vehicle_h[d_num])
                # 트래커 활성화
                tracker = cv2.TrackerCSRT_create()
                tracker.init(Substracted, initBB)  # 트래커를 원본이미지가 아닌  백그라운드 Substracted 된 이미지에서 트래킹함


    # 트래커 활성화시 동작
    if initBB is not None:
        # grab the new bounding box coordinates of the object
        (success, box) = tracker.update(Substracted)

        # check to see if the tracking was a success
        # 트래킹 성공시
        if success:
            (x, y, w, h) = [int(v) for v in box]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.rectangle(Substracted, (x, y), (x + w, y + h), (0, 255, 255), 2)

            Tracking_Xp = x + int(w/2)
            Tracking_Yp = y + h

            # Tracking Point 와 Detected Point의 거리가 150픽셀 이하인 경우 매칭및 트래커 박스 재조정
            Matched = False
            Matched_Xp = 0
            Matched_Yp = 0

            for i in range(0, len(Vehicle_x), 1):

                Vehicle_Xp = Vehicle_x[i] + int(Vehicle_w[i] / 2)
                Vehicle_Yp = Vehicle_y[i] + Vehicle_h[i]

                #트래커 포인트와 디텍팅 포인트와의 거리가 150이하인 경우
                if int(math.sqrt(pow(abs(Tracking_Xp-Vehicle_Xp), 2) + pow(abs(Tracking_Yp-Vehicle_Yp), 2))) < 200:
                    cv2.line(frame, (Tracking_Xp, Tracking_Yp),
                             (Vehicle_Xp, Vehicle_Yp), (125,255,125), 2)

                    Matched = True
                    Matched_Xp=Vehicle_Xp
                    Matched_Yp=Vehicle_Yp

                    #트래커의 박스를 재조정하기위한 Bounding box
                    tempBB= (Vehicle_x[i], Vehicle_y[i], Vehicle_w[i], Vehicle_h[i])

                    #트래커삭제 및 갱신
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(Substracted, tempBB)  # 트래커를 원본이미지가 아닌  백그라운드 Substracted 된 이미지에서 트래킹함
                    break

            #매칭이 트루이고, 매칭된 디텍티드 포인트가 영역밖에 존재하는 경우 - 삭제
            if (Matched == True):

                initBB_xy = (initBB[0] + int(initBB[2] / 2), initBB[1] + initBB[3])

                Matched_xy =(Matched_Xp,Matched_Yp)

                RED_line_start_xy = (vertices[0][0][0],vertices[0][0][1])
                RED_line_end_xy = (vertices[0][3][0],vertices[0][3][1])

                BLUE_line_start_xy = (vertices[0][1][0],vertices[0][1][1])
                BLUE_line_end_xy =(vertices[0][2][0],vertices[0][2][1])

                #tracker에서 매칭된 디텍티드 포인트로 변경하여 주석처리


                if intersect(initBB_xy, Matched_xy, RED_line_start_xy, RED_line_end_xy):
                    RED_cnt = RED_cnt + 1
                    # initBB,lastBB, tracker 초기화
                    cv2.line(frame, (initBB[0] + int(initBB[2] / 2), initBB[1] + initBB[3]),
                             (Matched_Xp, Matched_Yp), COLOR_RED, 2)
                    initBB = None
                    tracker = cv2.TrackerCSRT_create()

                if intersect(initBB_xy, Matched_xy, BLUE_line_start_xy, BLUE_line_end_xy):
                    BLUE_cnt = BLUE_cnt + 1
                    cv2.line(frame, (initBB[0] + int(initBB[2] / 2), initBB[1] + initBB[3]),
                             (Matched_Xp, Matched_Yp), COLOR_RED, 2)
                    # initBB,lastBB, tracker 초기화
                    initBB = None
                    tracker = cv2.TrackerCSRT_create()

    return tracker, initBB,RED_cnt, BLUE_cnt


def draw_line(frame, vertices, RED_cnt, BLUE_cnt):
    # Red_Line
    cv2.line(frame, (vertices[0][0][0], vertices[0][0][1]), (vertices[0][3][0], vertices[0][3][1]), (0, 0, 255), 2)
    cv2.putText(frame, "IN : " + str(RED_cnt), (vertices[0][0][0], vertices[0][0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Blue_Line
    cv2.line(frame, (vertices[0][1][0], vertices[0][1][1]), (vertices[0][2][0], vertices[0][2][1]), (255, 0, 0), 2)
    cv2.putText(frame, "OUT : " + str(BLUE_cnt), (vertices[0][1][0], vertices[0][1][1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)


def detecting_zone(vertices):
    # Detecting Zone
    pts = np.array([[vertices[0][1][0] + int(2 / 3 * (vertices[0][0][0] - vertices[0][1][0])), vertices[0][0][1] + int(1 / 3 * (vertices[0][1][1] - vertices[0][0][1]))], \
                      [vertices[0][1][0] + int(1 / 3 * (vertices[0][0][0] - vertices[0][1][0])), vertices[0][0][1] + int(2 / 3 * (vertices[0][1][1] - vertices[0][0][1]))], \
                      [vertices[0][3][0] + int(2 / 3 * (vertices[0][2][0] - vertices[0][3][0])), vertices[0][3][1] + int(2 / 3 * (vertices[0][2][1] - vertices[0][3][1]))], \
                      [vertices[0][3][0] + int(1 / 3 * (vertices[0][2][0] - vertices[0][3][0])), vertices[0][3][1] + int(1 / 3 * (vertices[0][2][1] - vertices[0][3][1]))]], \
                     np.int32)
    return pts

def ccw(A,B,C):
    return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])

def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

if __name__ == '__main__':
    main()


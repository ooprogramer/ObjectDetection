
import cv2
import numpy as np

cap = cv2.VideoCapture("../video/19.mp4")

first_frame = cv2.imread("image/19_park.png")
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
first_gray = cv2.GaussianBlur(first_gray, (5, 5), 0)

pts1 = np.array([[1440,490],[1920,490],[1920,620],[1570,620]],np.int32)
#pts2 = np.array([[80,400],[530,400],[420,520],[100,500]],np.int32)
#pts3 = np.array([[140,340],[600,330],[530,400],[80,400],[80,375]],np.int32)
while True:
    _, frame = cap.read()


    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    difference = cv2.absdiff(first_gray, gray_frame)
    _, difference = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)

    cv2.polylines(difference, [pts1], True, (255,255,255), 2)###########polylines
    #cv2.polylines(difference, [pts2], True, (255,255,255), 2)
    #cv2.polylines(difference, [pts3], True, (255,255,255), 2)

    #first_frame = cv2.resize(first_frame, (1280, 720), interpolation=cv2.INTER_CUBIC)
    frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_CUBIC)
    difference = cv2.resize(difference, (1280, 720), interpolation=cv2.INTER_CUBIC)
    #cv2.imshow("First frame", first_frame)
    #cv2.imshow("Frame", frame)

   # difference = cv2.dilate(difference, (25 ,25), 50)
    mask3 = cv2.cvtColor(difference, cv2.COLOR_GRAY2BGR)  # 3 channel mask
    im_thresh_color = cv2.bitwise_and(frame, mask3)

    cv2.imshow("difference", difference)
    cv2.imshow("im_thresh_color", im_thresh_color)


    key = cv2.waitKey(30)
    if key == ord("q"):
        break

    if key == ord("s"):
        cv2.imwrite("output.png",frame)
        break

cap.release()
cv2.destroyAllWindows()

"""

import cv2
import numpy as np

cap = cv2.VideoCapture("../video/19.mp4")

subtractor = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=25, detectShadows=True)

while True:
    _, frame = cap.read()

    mask = subtractor.apply(frame)

    mask = cv2.resize(mask, (1280, 720), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("Frame", frame)
    cv2.imshow("mask", mask)

    key = cv2.waitKey(30)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
"""

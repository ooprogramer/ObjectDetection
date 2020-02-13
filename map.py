import cv2
import csv
import time

def main():
	frame = cv2.imread("image/map.png")
	frame = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_CUBIC)
	area_19 = _24_25(frame); loc_19 = 'down'
	area_22 = _28_29(frame); loc_22 = 'down'
	while True:
		f_19 = open('csv/cam19.csv', 'r')
		f_22 = open('csv/cam22.csv', 'r')
		rdr_19 = csv.reader(f_19)
		rdr_22 = csv.reader(f_22)
		for line in rdr_19:
			for i in range(0,4):
				car = line[0].split(' ')[i]
				count(frame,i,car,area_19,loc_19)
		for line in rdr_22:
			for i in range(0,4):
				car = line[0].split(' ')[i]
				count(frame,i,car,area_22,loc_22)

		cv2.imshow('map', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	f_19.close()
	f_22.close()


#left-up -> left-down -> right-up -> right-down

# for big frame
def _24_25(frame):
	#width = 47, height = 48, length = 61
	A24 = [(116,124), (163,172), 'A24'] 
	A25 = [(177,124), (224,172), 'A25'] 
	B24 = [(116,222), (163,270), 'B24'] 
	B25 = [(177,222), (224,270), 'B25'] 
	C24 = [(116,271), (163,319), 'C24']
	C25 = [(177,271), (224,319), 'C25']
	D24 = [(116,369), (163,417), 'D24']
	D25 = [(177,369), (224,417), 'D25']
	down = [C25, C24, D25, D24]
	up = [A25, A24, B24, B25]
	area = [up, down]

	nemo(frame, up, down)
	return area

def _28_29(frame):
	#width = 47, height = 48, length = 61
	A28 = [(357,124), (404,172), 'A28'] 
	A29 = [(418,124), (465,172), 'A29'] 
	B28 = [(357,222), (404,270), 'B28'] 
	B29 = [(418,222), (465,270), 'B29'] 
	C28 = [(357,271), (404,319), 'C28']
	C29 = [(418,271), (465,319), 'C29']
	D28 = [(357,369), (404,417), 'D28']
	D29 = [(418,369), (465,417), 'D29']
	down = [C29, C28, D29, D28]
	up = [A29, A28, B29, B28]
	area = [up, down]

	nemo(frame, up, down)
	return area


"""
## for small frame
def _24_25(frame):
	#width = 20, height = 22, length = 27
	A24 = [(50,53), (70,75), 'A24'] 
	A25 = [(77,53), (97,75), 'A25'] 
	B24 = [(50,96), (70,118), 'B24'] 
	B25 = [(77,96), (97,118), 'B25'] 
	C24 = [(50,117), (70,139), 'C24']
	C25 = [(77,117), (97,139), 'C25']
	D24 = [(50,160), (70,182), 'D24']
	D25 = [(77,160), (97,182), 'D25']
	down = [C25, C24, D25, D24]
	up = [A25, A24, B24, B25]
	area = [up, down]

	nemo(frame, up, down)
	return area

def _28_29(frame):
	#width = 20, height = 22, length = 27
	A28 = [(154,53), (174,75), 'A28'] 
	A29 = [(181,53), (201,75), 'A29'] 
	B28 = [(154,96), (174,118), 'B28'] 
	B29 = [(181,96), (201,118), 'B29'] 
	C28 = [(154,117), (174,139), 'C28']
	C29 = [(181,117), (201,139), 'C29']
	D28 = [(154,160), (174,182), 'D28']
	D29 = [(181,160), (201,182), 'D29']
	down = [C29, C28, D29, D28]
	up = [A29, A28, B29, B28]
	area = [up, down]

	nemo(frame, up, down)
	return area
"""

def nemo(frame, up, down):
	for i in down:
		sub_nemo(frame, i)
		text_down(i, frame)
	for i in up:
		sub_nemo(frame, i)
		text_up(i, frame)

def text_down(space, frame):
	cv2.putText(frame, str(space[2]), (space[1][0]-30, space[1][1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) #-> big
	#cv2.putText(frame, str(space[2]), (space[1][0]-20, space[1][1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1) #-> small

def text_up(space, frame):
	cv2.putText(frame, str(space[2]), (space[0][0], space[0][1]-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) #-> big
	#cv2.putText(frame, str(space[2]), (space[0][0], space[0][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1) #-> small

def sub_nemo(frame, rec):
	cv2.rectangle(frame, rec[0], rec[1], (0,0,0), -1)
	cv2.rectangle(frame, rec[0], rec[1], (200,200,200), 2)

def count(frame, i, car, area, loc):
	up_down = 0
	if loc == 'down':
		up_down = 1
	abcd = area[up_down]

	if car == '3':
		cv2.rectangle(frame, (abcd[i][0][0]+4, abcd[i][0][1]+4), (abcd[i][1][0]-4, abcd[i][1][1]-4), (200,200,200), -1)
	elif car == '2': # small -> 13 / big -> 32
		sub_nemo(frame, abcd[i])
		cv2.rectangle(frame, (abcd[i][0][0]+4, abcd[i][0][1]+4), (abcd[i][0][0]+31, abcd[i][1][1]-4), (200,200,200), -1)
	elif car == '1': # small -> 7 / big -> 16
		sub_nemo(frame, abcd[i])
		cv2.rectangle(frame, (abcd[i][0][0]+4, abcd[i][0][1]+4), (abcd[i][0][0]+15, abcd[i][1][1]-4), (200,200,200), -1)

if __name__ == '__main__':
    time.sleep(1)
    main()

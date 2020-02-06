import cv2
import cam19, cam22

lot = cv2.imread("image/map.png")

def main():
	global lot

	_24_25()
	_28_29()

	lot = cv2.resize(lot, (960, 540), interpolation=cv2.INTER_CUBIC)
	cv2.imshow('map', lot)
	cv2.waitKey(0)

def _24_25():
	#width = 20, height = 22, length = 27
	A24 = [(50,53), (70,75), 'A24'] 
	A25 = [(77,53), (97,75), 'A25'] 
	B24 = [(50,96), (70,118), 'B24'] 
	B25 = [(77,96), (97,118), 'B25'] 
	C24 = [(50,117), (70,139), 'C24']
	C25 = [(77,117), (97,139), 'C25']
	D24 = [(50,160), (70,182), 'D24']
	D25 = [(77,160), (97,182), 'D25']
	down = [C24, D24, C25, D25]
	up = [A24, B24, A25, B25]

	nemo(lot, up, down)

def _28_29():
	#width = 20, height = 22, length = 27
	A28 = [(181,53), (201,75), 'A28'] 
	A29 = [(208,53), (228,75), 'A29'] 
	B28 = [(181,96), (201,118), 'B28'] 
	B29 = [(208,96), (228,118), 'B29'] 
	C28 = [(181,117), (201,139), 'C28']
	C29 = [(208,117), (228,139), 'C29']
	D28 = [(181,160), (201,182), 'D28']
	D29 = [(208,160), (228,182), 'D29']
	down = [C28, D28, C29, D29]
	up = [A28, B28, A29, B29]

	nemo(lot, up, down)



def nemo(frame, up, down):
	for i in down:
		cv2.rectangle(frame, i[0], i[1], (255,255,255), 1)
		text_down(i, frame)
	for i in up:
		cv2.rectangle(frame, i[0], i[1], (255,255,255), 1)
		text_up(i, frame)

def text_down(space, frame):
	cv2.putText(frame, str(space[2]), (space[1][0]-20, space[1][1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

def text_up(space, frame):
	cv2.putText(frame, str(space[2]), (space[0][0], space[0][1]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

if __name__ == '__main__':
    main()

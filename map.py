import cv2
import os.path

FIFO_FILENAME = './cam19'

def main():
	frame = cv2.imread("image/map.png")
	#if os.path.exists(FIFO_FILENAME):
	fp_fifo = open(FIFO_FILENAME, "r")
	print('1')
	data = fp_fifo.read()
	print('2')
	print(data)

	_24_25(frame)
	_28_29(frame)
	frame = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_CUBIC)

	while True:
		cv2.imshow('map', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	


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
	down = [C24, D24, C25, D25]
	up = [A24, B24, A25, B25]

	nemo(frame, up, down)

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
	down = [C28, D28, C29, D29]
	up = [A28, B28, A29, B29]

	nemo(frame, up, down)



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

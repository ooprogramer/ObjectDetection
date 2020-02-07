import os.path
FIFO_FILENAME = './fifo-test'
r_up=2; r_down=5; l_up=8; l_down=7;
park_cnt = [l_up,l_down,r_up,r_down]

if os.path.exists(FIFO_FILENAME):
    fp_fifo = open(FIFO_FILENAME, "w")
else:
    os.mkfifo(FIFO_FILENAME)
    fp_fifo = open(FIFO_FILENAME, "w")

for i in range(5):
    fp_fifo.write(str(park_cnt))
    fp_fifo.write("\n")



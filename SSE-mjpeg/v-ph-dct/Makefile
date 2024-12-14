CC = gcc
CFLAGS = -Wall -g -O3 -msse4.1
LDFLAGS = -lm -lc

all: mjpeg_encoder

mjpeg_encoder: mjpeg_encoder.o

clean:
	rm -f *.o mjpeg_encoder
	rm -f test.mjpeg
	rm -f *~

foreman: 
	./mjpeg_encoder -w 352 -h 288 -f 30 -o test.mjpeg ../foreman.yuv
play:
	mplayer test.mjpeg

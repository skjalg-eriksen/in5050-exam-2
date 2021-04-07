# Acknowledgement: Functionality for creating make rules of dependencies is
# based on code presented here <http://codereview.stackexchange.com/q/11109>

# CC = gcc  # To specify compiler use: '$ CC=clang make clean all'
#CFLAGS = -O1 -fno-tree-vectorize -march=armv8.2-a -Wall -g

CC = nvcc
CFLAGS = -O1 \
          -gencode arch=compute_72,code=sm_72 \
	  -gencode arch=compute_72,code=compute_72
CUFLAGS = -G -g -rdc=true \
          -gencode arch=compute_72,code=sm_72 \
	  -gencode arch=compute_72,code=compute_72

LDFLAGS= -lm -gencode arch=compute_72,code=sm_72 \
          -gencode arch=compute_72,code=compute_72 -lnvToolsExt
#LDFLAGS = -lm
# Use the compiler to generate make rules. See gcc manual for details.
#MFLAGS = -MMD -MP -MF

SOURCES = $(wildcard *.c)
SOURCES_cuda = $(wildcard *.cu)
OBJECTS = $(SOURCES:.c=.o)
OBJECTS_cuda = $(SOURCES_cuda:.cu=.o)
DEPENDENCIES = $(addprefix .,$(SOURCES:.c=.d))  # Add dot prefix to hide files.

.PHONY: clean  all

all: c63enc c63dec c63pred

%.o: %.cu
	$(CC) $(CFLAGS) $(CUFLAGS) -c $<

c63enc: c63enc.o dsp.o tables.o io.o c63_write.o common.o me.o
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@
c63dec: c63dec.c dsp.o tables.o io.o common.o me.o
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@
c63pred: c63dec.c dsp.o tables.o io.o common.o me.o
	$(CC) $^ -DC63_PRED $(CFLAGS) $(LDFLAGS) -o $@

#%.o: %.c
#	$(CC) $(CFLAGS) $(MFLAGS) $(addprefix .,$(patsubst %.o,%.d,$@)) -c $< -o $@

clean:
	$(RM) c63enc c63dec c63pred $(OBJECTS) $(OBJECTS_cuda) $(DEPENDENCIES)

-include $(DEPENDENCIES)

test:
	nvprof ./c63enc -h 288 -w 352 -o ../test.c63 -f 5 /mnt/sdcard/cipr/foreman.yuv

test2:
	nvprof --unified-memory-profiling off ./c63enc -h 288 -w 352 -o ../test.c63 -f 5 /mnt/sdcard/cipr/foreman.yuv

test3:
	nvprof --unified-memory-profiling off ./c63enc -h 288 -w 352 -o ../test.c63 /mnt/sdcard/cipr/foreman.yuv

run:
	./c63enc -h 288 -w 352 -o ../test1.c63 -f 5 ../../mjpeg/foreman.yuv

run2:
	./c63enc -h 288 -w 352 -o ../test1.c63 ../../mjpeg/foreman.yuv
debug:
		cuda-gdb ./c63enc --args -h 288 -w 352 -o ../test.c63 -f 5 ../../mjpeg/foreman.yuv
	#/mnt/sdcard/cipr/foreman.yuv

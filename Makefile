
# MAC M1 -- installed opencv package via brew install opencv
# LDFLAGS:=$(shell pkg-config --cflags --libs /opt/homebrew/Cellar/opencv/*/lib/pkgconfig/opencv4.pc)

# INTEL MAC
# LDFLAGS:=$(shell pkg-config --cflags --libs /usr/local/Cellar/opencv/*/lib/pkgconfig/opencv4.pc)

# CIMS Server, need to load module
# module load gcc-12.2
OPENCV_CFLAGS := $(shell pkg-config --cflag opencv)
OPENCV_LIBS := $(shell pkg-config --libs opencv)
LDFLAGS := $(OPENCV_LIBS) $(OPENCV_CFLAGS)

# CIMS Cuda Server
CUDA_INCDIR = -I $(CUDA_HOME)/include -I $(CUDA_HOME)/samples/common/inc
CUDA_LIBS = -lblas -L${CUDA_HOME}/lib64 -lcudart

NVCC = nvcc
NVCCFLAGS = -std=c++11
NVCCFLAGS += -Xcompiler "-fopenmp" # pass -fopenmp to host compiler (g++)
NVCCFLAGS += --expt-relaxed-constexpr

#NVCCFLAGS += --gpu-architecture=compute_35 --gpu-code=compute_35
#NVCCFLAGS += --gpu-architecture=compute_60 --gpu-code=compute_60 # specify Pascal architecture
#NVCCFLAGS += -Xptxas -v # display compilation summary

TARGETS = $(basename $(wildcard *.cu))

CC:=g++
CFLAGS:=--std=c++14 -ggdb
SRC_FILES:=$(wildcard ./*.cpp)
OBJ_FILES:=$(patsubst %.cpp,obj/%.o,$(SRC_FILES))

#all: burningship
all : $(TARGETS)

burningship: $(OBJ_FILES)
	$(CC) $(LDFLAGS) -o $@ $^

obj/%.o: %.cpp
	$(CC) $(LDFLAGS) $(CFLAGS) -c -o $@ $<

cuda:
	$(NVCC) $(LDFLAGS) $(NVCCFLAGS) burningship_gpu.cu -o burningship_gpu

clean:
	rm burningship
	rm obj/*.o

CC:=g++

# MAC M1 -- installed opencv package via brew install opencv
LDFLAGS:=$(shell pkg-config --cflags --libs /opt/homebrew/Cellar/opencv/*/lib/pkgconfig/opencv4.pc)

# INTEL MAC
#LDFLAGS:=$(shell pkg-config --cflags --libs /usr/local/Cellar/opencv/*/lib/pkgconfig/opencv4.pc)

CFLAGS:=--std=c++14 -ggdb
SRC_FILES:=$(wildcard ./*.cpp)
OBJ_FILES:=$(patsubst %.cpp,obj/%.o,$(SRC_FILES))

all: mandelbrot

mandelbrot: $(OBJ_FILES)
	$(CC) $(LDFLAGS) -o $@ $^

obj/%.o: %.cpp
	$(CC) $(LDFLAGS) $(CFLAGS) -c -o $@ $<

clean:
	rm mandelbrot
	rm obj/*.o

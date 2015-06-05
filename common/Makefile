
include ./Makefile.config

CXX = g++
CXX_FLAGS = -O3 \
						$(INCLUDE_DIRS) \
						-fpermissive \
						-I./include/ \
						-I$(CAFFE)/include/ \
						-I$(CUDA)/include/

SRC=src

TARGET = tonic

$(SRC)/tonic.o: $(SRC)/tonic.cpp $(SRC)/socket.cpp
	$(CXX) -c $(CXX_FLAGS) $< -o $@

all:
	ar rcs libtonic.a $(SRC)/tonic.o

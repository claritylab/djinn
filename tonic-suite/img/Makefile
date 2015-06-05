CAFFE=/home/jahausw/projects/sirius-caffe/distribute
CUDA=/usr/local/cuda
COMMON=../../common
CXX = g++
CXX_FLAGS  = -O3 \
						 -fpermissive \
						 -I$(COMMON)/include \
						 -I$(CAFFE)/include \
						 -I$(CUDA)/include \
						 -DCPU_ONLY

LINK_FLAGS = -lopencv_core \
						 -lopencv_highgui \
						 -lopencv_imgproc \
						 -lopencv_objdetect \
						 ../../common/libsocket.a \
						 libflandmark_static.a \
						 -lboost_program_options \
						 -lboost_filesystem \
						 -lboost_system \
						 -lglog \
						 -lrt \
						 -lpthread \
						 $(CAFFE)/lib/libcaffe.so

SRC=src

# File names
EXEC = tonic-img
SOURCES = $(wildcard $(SRC)/*.cpp) $(COMMON)/src/tonic.cpp
OBJECTS = $(SOURCES:.cpp=.o)

# Main target
$(EXEC): $(OBJECTS) Makefile
	$(CXX) $(OBJECTS) -o $(EXEC) $(LINK_FLAGS)

# To obtain object files
%.o: %.cpp Makefile
	$(CXX) -c $(CXX_FLAGS) $(EXTRA_FLAGS) $< -o $@

# To remove generated files
clean:
	rm -f $(EXEC) $(SRC)/*.o $(COMMON)/src/*.o

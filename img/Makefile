OPENCV=/home/jahausw/opencv/opencv-2.4.9/build
CAFFE=/usr/local/include/caffe
CUDA=/usr/local/cuda
CXX = g++
CXX_FLAGS = -fpermissive -I$(CAFFE) -I$(CUDA)/include -I$(OPENCV)/include
LINK_FLAGS = $(OPENCV)/lib/libopencv_core.so \
			 $(OPENCV)/lib/libopencv_highgui.so \
			 $(OPENCV)/lib/libopencv_imgproc.so \
			 $(OPENCV)/lib/libopencv_objdetect.so \
			 ../libsocket.a libflandmark.a \
			 -lboost_program_options -lboost_filesystem -lboost_system \
			 -lglog -lrt -lpthread

SRC=src

# File names
EXEC = img-client
SOURCES = $(wildcard $(SRC)/*.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

# Main target
$(EXEC): $(OBJECTS) Makefile
	$(CXX) $(OBJECTS) -o $(EXEC) $(LINK_FLAGS)

# To obtain object files
%.o: %.cpp Makefile
	$(CXX) -c $(CXX_FLAGS) $(EXTRA_FLAGS) $< -o $@

# To remove generated files
clean:
	rm -f $(EXEC) $(SRC)/*.o

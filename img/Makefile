CAFFE=/home/jahausw/projects/caffe/distribute
CUDA=/usr/local/cuda
CXX = g++
CXX_FLAGS  = -fpermissive -I$(CAFFE)/include -I$(CUDA)/include
LINK_FLAGS = -lopencv_core \
			 -lopencv_highgui \
			 -lopencv_imgproc \
			 -lopencv_objdetect \
			 -lcaffe \
			 ../libsocket.a \
			 libflandmark.a \
			 -lboost_program_options \
			 -lboost_filesystem \
			 -lboost_system \
			 -lglog \
			 -lrt \
			 -lpthread

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

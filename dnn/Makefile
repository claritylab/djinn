CAFFE=/home/jahausw/projects/caffe/distribute
CUDA=/usr/local/cuda
CXX = g++
CXX_FLAGS = -w -fpermissive -I$(CAFFE)/include -I$(CUDA)/include -g
LINK_FLAGS = -lpthread -ldl $(CAFFE)/lib/libcaffe.so $(CUDA)/lib64/libcudart.so ../libsocket.a -lrt -lboost_program_options -lboost_filesystem -lboost_system -lglog

SRC=src

# File names
EXEC = dnn-server
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

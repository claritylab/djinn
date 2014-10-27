CAFFE=/home/ypkang/caffe/distribute
CUDA=/usr/local/cuda
CXX = g++
CXX_FLAGS = -w -fpermissive -I$(CAFFE)/include -I$(CUDA)/include -g
LINK_FLAGS = -pthread -ldl $(CAFFE)/lib/libcaffe.so ../libsocket.a -lrt -lpthread -lboost_program_options -lboost_filesystem -lboost_system -lglog
# LINK_FLAGS = $(CAFFE)/lib/libcaffe.so -lrt -lpthread -lboost_program_options -lboost_filesystem -lboost_system -lglog

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

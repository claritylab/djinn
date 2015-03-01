CAFFE=/home/ypkang/caffe/distribute/
CUDA=/usr/local/cuda
CXX = g++
CXX_FLAGS = -w -fpermissive -I$(CAFFE)/include -I$(CUDA)/include -g
LINK_FLAGS = -L/$(CUDA)/lib -lcudart -lpthread -ldl ../libsocket.a -lrt -lboost_timer -lboost_chrono -lboost_program_options -lboost_filesystem -lboost_system -lglog -lrt

SRC=src

# File names
EXEC = dnn-server
SOURCES = $(wildcard $(SRC)/*.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

all : dnn-server 

dnn-server: $(OBJECTS) Makefile
	$(CXX) $(OBJECTS) -o dnn-server $(LINK_FLAGS) /home/ypkang/caffe/distribute/lib/libcaffe.so

# To obtain object files
%.o: %.cpp Makefile
	$(CXX) -c $(CXX_FLAGS) $< -o $@

# To remove generated files
clean:
	rm -f dnn-server $(SRC)/*.o

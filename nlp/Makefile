CAFFE=/home/jahausw/projects/caffe/distribute
CUDA=/usr/local/cuda
CXX = g++
CXX_FLAGS = -w -fpermissive -I$(CAFFE)/include -I$(CUDA)/include -g
LINK_FLAGS = -lpthread -ldl $(CAFFE)/lib/libcaffe.so -lrt -lglog -lthrift -lgflags

SRC=src

# File names
EXEC = nlp-client
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

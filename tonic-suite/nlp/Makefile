#CAFFE=/home/jahausw/projects/sirius-caffe/distribute
CAFFE=/home/jahausw/projects/caffe/distribute
CUDA=/usr/local/cuda
COMMON=../../common
CXX = g++
CXX_FLAGS  = -O3 \
						 -fpermissive \
						 -I$(COMMON)/include \
						 -I$(CAFFE)/include \
						 -I$(CUDA)/include \
						 -DCPU_ONLY

LINK_FLAGS = -L/usr/local/lib ../../common/libsocket.a \
						 -lboost_program_options \
						 -lboost_filesystem \
						 -lboost_system \
						 -lglog \
						 -lrt \
						 -lpthread \
						 $(CAFFE)/lib/libcaffe.so

SRC=src

# File names
EXEC = tonic-nlp
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

CAFFE=/home/jahausw/projects/sirius-caffe/distribute
CUDA=/usr/local/cuda
COMMON=../common

CXX = g++
CXX_FLAGS = -O3 \
						-fpermissive \
						-I$(COMMON)/include \
						-I$(CAFFE)/include \
						-I$(CUDA)/include \
						-DCPU_ONLY

LINK_FLAGS = -lpthread \
						 -ldl \
						 -lrt \
						 -lboost_timer \
						 -lboost_chrono \
						 -lboost_program_options \
						 -lboost_filesystem \
						 -lboost_system \
						 -lglog \
						 $(COMMON)/libsocket.a \
						 $(CAFFE)/lib/libcaffe.so

SRC=src

# File names
EXEC = djinn
SOURCES = $(wildcard $(SRC)/*.cpp) $(COMMON)/src/tonic.cpp
OBJECTS = $(SOURCES:.cpp=.o)

$(EXEC): $(OBJECTS) Makefile
	$(CXX) $(OBJECTS) -o $(EXEC) $(LINK_FLAGS)

# To obtain object files
%.o: %.cpp Makefile
	$(CXX) -c $(CXX_FLAGS) $< -o $@

# To remove generated files
clean:
	rm -f $(EXEC) $(SRC)/*.o $(COMMON)/src/*.o

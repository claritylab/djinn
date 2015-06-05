COMMON=../common
include $(COMMON)/Makefile.config

CXX = g++
CXX_FLAGS  = -O3 \
						 $(INCLUDE_DIRS) \
						 -fpermissive \
						 -I$(COMMON)/include \
						 -I$(CAFFE)/include \
						 -I$(CUDA)/include

ifeq ($(CPU_ONLY), 1)
	CXX_FLAGS  += -DCPU_ONLY
endif

LINK_FLAGS = $(LIBRARY_DIRS) \
						 -lpthread \
						 -ldl \
						 -lrt \
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

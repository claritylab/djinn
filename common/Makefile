
include ./Makefile.config

CXX = g++
CXX_FLAGS = -O3 \
						$(INCLUDE_DIRS) \
						-fpermissive \
						-I./include/ \
						-I$(CAFFE)/include/ \
						-I$(CUDA)/include/

LINKFLAGS = $(LIBRARY_DIRS) \
						-lboost_program_options \
						-lboost_filesystem \
						-lboost_system \
						-lglog \
						-lrt \
						-lpthread \
						$(CAFFE)/lib/libcaffe.so
SRC=src
TARGET = libtonic.a

SOURCES = $(wildcard $(SRC)/*.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

$(TARGET): $(OBJECTS) Makefile
	ar rcs $(TARGET) $(OBJECTS)

# To obtain object files
%.o: %.cpp Makefile
	$(CXX) -c $(CXX_FLAGS) $< -o $@

clean:
	rm libtonic.a $(SRC)/tonic.o

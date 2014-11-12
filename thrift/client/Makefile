# cuda path
CUDA=/usr/local/cuda
# caffe path
PATH_CAFFE = /usr/local/include/caffe
# folly path
PATH_FOLLY = /usr/local/include/folly/
# double-conversion path
PATH_DBCON = /usr/local/include/double-conversion
# thrift
PATH_THRIFT = /usr/local/include/thrift

CXX = g++

INCLUDES   = -I$(PATH_DBCON) \
			 -I$(PATH_FOLLY) \
			 -I$(PATH_CAFFE) \
			 -I$(CUDA)/include \

LIBRARIES  = -levent \
			 -lfolly \
			 -lgflags \
			 -lglog \
			 -lpthread \
			 -lrt \
			 -lboost_system \
			 -lthriftcpp2 \
			 -lcaffe \
			 -L$(PATH_DBCON) \
			 -L$(CUDA)/lib64 \
			 -L/usr/local/lib/ \

CXX_CFLAGS = -std=c++11 -Wall -D_GNU_SOURCE $(INCLUDES) -fpermissive
CXX_LFLAGS = $(LIBRARIES)
CXX_DFLAGS = -DDEBUG -g

EXEC = DnnClient
SRC = src
SOURCES = $(wildcard $(SRC)/*.cpp) ../gen-cpp2/Dnn.cpp ../gen-cpp2/App.cpp ../gen-cpp2/dnn_types.cpp
OBJECTS = $(SOURCES:.cpp=.o)

all: $(EXEC)

debug: CXX += $(CXX_DFLAGS)
debug: $(EXEC)

# Main target
$(EXEC): $(OBJECTS) Makefile
	$(CXX) $(OBJECTS) -o $(EXEC) $(CXX_LFLAGS)

%.o: %.cpp Makefile
	$(CXX) -c $(CXX_CFLAGS) $(CXX_LFLAGS) $< -o $@ -lpthread

clean:
	$(RM) src/*.o ../gen-cpp2/*.o *~ $(BINARY)

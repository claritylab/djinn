CAFFE=/home/ypkang/git/caffe/distribute
CUDA=/usr/local/cuda

# folly path
PATH_FOLLY = /home/yunqi/folly/
# double-conversion path
PATH_DBCON = /home/yunqi/double-conversion/

CXX = g++

INCLUDES   = -I$(PATH_DBCON)src -I$(PATH_FOLLY) -I$(CAFFE)/include -I$(CUDA)/include -g
LIBRARIES  = -levent -lfolly -lgflags -lglog -lpthread -lrt -lboost_system \
						 -lthriftcpp2 \
             -L$(PATH_DBCON) -L/usr/local/lib/ \
             $(CAFFE)/lib/libcaffe.so

CXX_CFLAGS = -std=c++11 -Wall -D_GNU_SOURCE $(INCLUDES)
CXX_LFLAGS = $(LIBRARIES)
CXX_DFLAGS = -DDEBUG -g

DNN_SRCS = DnnServer.cpp \
						 DnnHandler.cpp \
						 ../gen-cpp2/Dnn.cpp

SRCS = $(DNN_SRCS)

BINARY = DnnServer

all: $(BINARY)

debug: CXX += $(CXX_DFLAGS)
debug: $(BINARY)

$(BINARY): $(SRCS)
	$(CXX) $(CXX_CFLAGS) $(CXX_LFLAGS) -o $@ $?

clean:
	$(RM) *.o *~ $(BINARY)

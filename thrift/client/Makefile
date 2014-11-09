CUDA=/usr/local/cuda

# caffe path
PATH_CAFFE = /usr/local/include/caffe
# folly path
PATH_FOLLY = /usr/local/include/folly/
# double-conversion path
# PATH_DBCON = /usr/local/include/double-conversion
PATH_DBCON = /home/yunqi/double-conversion/
# thrift
PATH_THRIFT = /usr/local/include/thrift

CXX = g++

INCLUDES   =  -I$(PATH_DBCON)src \
			  -I$(PATH_FOLLY) \
			  -I$(PATH_CAFFE) \
			  -I$(PATH_THRIFT) \
			  -I$(CUDA)/include \

LIBRARIES  = -levent \
			 -lfolly \
			 -lgflags \
			 -lglog \
			 -lpthread \
			 -lrt \
			 -lboost_system \
			 -lthriftcpp2 \
			 -L$(PATH_DBCON) \
			 -lcaffe \
			 -L/usr/local/lib/ \

CXX_CFLAGS = -std=c++11 -Wall -D_GNU_SOURCE $(INCLUDES)
CXX_LFLAGS = $(LIBRARIES)
CXX_DFLAGS = -DDEBUG -g

SRCS = DnnClient.cpp \
	   ../gen-cpp2/Dnn.cpp

BINARY = DnnClient

all: $(BINARY)

debug: CXX += $(CXX_DFLAGS)
debug: $(BINARY)

$(BINARY): $(SRCS)
	$(CXX) $(CXX_CFLAGS) $(CXX_LFLAGS) -o $@ $?

clean:
	$(RM) *.o *~ $(BINARY)

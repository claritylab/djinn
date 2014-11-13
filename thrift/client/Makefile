include ./kaldi.mk

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
# kaldi
PATH_KALDI = /home/ypkang/git/kaldi-trunk/src

CXX = g++

INCLUDES   = -I$(PATH_DBCON) \
			 -I$(PATH_FOLLY) \
			 -I$(PATH_CAFFE) \
			 -I$(PATH_KALDI) \
			 -I$(CUDA)/include \

STATICLIBS = ./kaldi-nnet.a \
						 ./kaldi-cudamatrix.a \
						 ./kaldi-lat.a \
						 ./kaldi-hmm.a \
						 ./kaldi-tree.a \
						 ./kaldi-matrix.a \
						 ./kaldi-util.a \
						 ./kaldi-base.a \

LIBRARIES  = -ldl \
			 -levent \
			 -lfolly \
			 -lgflags \
			 -lglog \
			 -lpthread \
			 -lrt \
			 -lboost_system \
			 -lthriftcpp2 \
			 -lcaffe \
			 -L$(CUDA)/lib64 \
			 -lcudart \
			 -lcublas \
			 -lcblas \
			 -lblas \
			 -lclapack \
			 -L/usr/local/lib/ \

CXX_CFLAGS = -std=c++11 -Wall -D_GNU_SOURCE $(INCLUDES) -fpermissive $(KALDICXXFLAGS)
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
	$(CXX) $(OBJECTS) -o $(EXEC) $(CXX_LFLAGS) $(STATICLIBS)

%.o: %.cpp Makefile
	$(CXX) -c $(CXX_CFLAGS) $(CXX_LFLAGS) $< -o $@ -lpthread

clean:
	$(RM) src/*.o ../gen-cpp2/*.o *~ $(BINARY)

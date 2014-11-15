CAFFE=/usr/local/include/caffe/
CUDA=/usr/local/cuda
CXX = g++
CXX_FLAGS = -w -fpermissive -I$(CAFFE) -I$(CUDA)/include -g
LINK_FLAGS = -L/$(CUDA)/lib64 -lcudart -lpthread -ldl ../libsocket.a -lrt -lboost_timer -lboost_chrono -lboost_program_options -lboost_filesystem -lboost_system -lglog -lrt

SRC=src

# File names
EXEC = dnn-server
SOURCES = $(wildcard $(SRC)/*.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

all : dnn-server \
	    dnn-server1 \
			dnn-server2 \
	    dnn-server3 \
	    dnn-server4 \
			dnn-server5 \
			dnn-server6 \
			dnn-server7

dnn-server: $(OBJECTS) Makefile
	$(CXX) $(OBJECTS) -o dnn-server $(LINK_FLAGS) /usr/local/lib/libcaffe0.so

dnn-server1: $(OBJECTS) Makefile
	$(CXX) $(OBJECTS) -o dnn-server1 $(LINK_FLAGS) /usr/local/lib/libcaffe1.so

dnn-server2: $(OBJECTS) Makefile
	$(CXX) $(OBJECTS) -o dnn-server2 $(LINK_FLAGS) /usr/local/lib/libcaffe2.so

dnn-server3: $(OBJECTS) Makefile
	$(CXX) $(OBJECTS) -o dnn-server3 $(LINK_FLAGS) /usr/local/lib/libcaffe3.so

dnn-server4: $(OBJECTS) Makefile
	$(CXX) $(OBJECTS) -o dnn-server4 $(LINK_FLAGS) /usr/local/lib/libcaffe4.so

dnn-server5: $(OBJECTS) Makefile
	$(CXX) $(OBJECTS) -o dnn-server5 $(LINK_FLAGS) /usr/local/lib/libcaffe5.so

dnn-server6: $(OBJECTS) Makefile
	$(CXX) $(OBJECTS) -o dnn-server6 $(LINK_FLAGS) /usr/local/lib/libcaffe6.so

dnn-server7: $(OBJECTS) Makefile
	$(CXX) $(OBJECTS) -o dnn-server7 $(LINK_FLAGS) /usr/local/lib/libcaffe7.so 

# To obtain object files
%.o: %.cpp Makefile
	$(CXX) -c $(CXX_FLAGS) $< -o $@

# To remove generated files
clean:
	rm -f dnn-server dnn-server1 dnn-server2 dnn-server3 dnn-server4 $(SRC)/*.o \
	dnn-server5 dnn-server6 dnn-server7 $(SRC)/*.o

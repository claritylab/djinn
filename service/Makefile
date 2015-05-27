CAFFE=/usr/local/include/caffe/
CUDA=/usr/local/cuda
CXX = g++
CXX_FLAGS = -w -fpermissive -I$(CAFFE) -I$(CUDA)/include -O3
LINK_FLAGS = -L/$(CUDA)/lib64 \
			 -lcudart \
			 -lpthread \
			 -ldl \
			 -lrt \
			 -lboost_timer \
			 -lboost_chrono \
			 -lboost_program_options \
			 -lboost_filesystem \
			 -lboost_system \
			 -lglog \
			 ../libsocket.a

SRC=src

# File names
EXEC = dnn-server
SOURCES = $(wildcard $(SRC)/*.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

$(EXEC): $(OBJECTS) Makefile
	$(CXX) $(OBJECTS) -o dnn-server $(LINK_FLAGS) /usr/local/lib/libcaffe0.so

# To obtain object files
%.o: %.cpp Makefile
	$(CXX) -c $(CXX_FLAGS) $< -o $@

# To remove generated files
clean:
	rm -f dnn-server $(SRC)/*.o

CXX = g++
CXX_FLAGS = -fpermissive
LINK_FLAGS = ../libsocket.a -lrt -lpthread -lboost_program_options -lboost_filesystem -lboost_system

SRC=src

# File names
EXEC = nlp-client
SOURCES = $(wildcard $(SRC)/*.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

# Main target
$(EXEC): $(OBJECTS) Makefile
	$(CXX) $(OBJECTS) -o $(EXEC) $(LINK_FLAGS)

# To obtain object files
%.o: %.cpp Makefile
	$(CXX) -c $(CXX_FLAGS) $(EXTRA_FLAGS) $< -o $@

# To remove generated files
clean:
	rm -f $(EXEC) $(SRC)/*.o

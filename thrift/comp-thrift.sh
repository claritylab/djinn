# compile .thrift into apps+server
thrift -r --gen cpp -out dnn/src dnn.thrift 
thrift -r --gen cpp -out img/src dnn.thrift 
thrift -r --gen cpp -out nlp-thrift/src dnn.thrift 

# this is annoying
rm -rf dnn/src/*.skeleton.cpp
rm -rf img/src/*.skeleton.cpp
rm -rf nlp-thrift/src/*.skeleton.cpp

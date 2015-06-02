DIR=..

g++ -O3 -c -o socket.o socket.cpp -fpermissive
ar rcs libsocket.a socket.o

cp libsocket.a $DIR

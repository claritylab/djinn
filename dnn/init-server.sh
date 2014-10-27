./dnn-server --service $1 \
             --portno $2 \
             --model net-configs/$1.prototxt \
             --weights weights/$1.dat \
             --debug 1

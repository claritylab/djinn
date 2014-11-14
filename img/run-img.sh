./img-client --hostname localhost \
             --portno $2 \
             --task $1 \
             --input input/$1-input.bin \
             --flandmark data/flandmark.dat \
             --haar data/haar.xml \
             --debug 1

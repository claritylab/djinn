./img-client --hostname 141.212.111.252 \
             --portno 8080 \
             --task $1 \
             --imcin input/$1-inputnet.prototxt \
             --flandmark data/flandmark.dat \
             --haar data/haar.xml \
             --debug 1

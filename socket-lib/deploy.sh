DIR=/home/ypkang/git/deep-ipa
KALDI_DIR=/home/ypkang/kaldi-trunk/src/nnetbin

cp libsocket.a $DIR
cp socket.h $DIR/img/src
cp socket.h $DIR/dnn/src
cp socket.h $DIR/nlp/src
cp libsocket.a $KALDI_DIR 
cp socket.h $KALDI_DIR

DIR=/home/jahausw/projects/clarity-coding/deep-ipa
KALDI_DIR=/home/jahausw/projects/dnn/kaldi-trunk/src/nnetbin

cp libsocket.a $DIR
cp socket.h $DIR/imc/src
cp socket.h $DIR/dnn/src
cp socket.h $DIR/nlp/src
cp libsocket.a $KALDI_DIR 

task=$1
numservers=$2
gpuid=$3
BASE_PORT=$(( 7999 + $gpuid*100 ))

for i in $(seq 1 $numservers);
do
  ./run-nlp.sh $1 $((BASE_PORT + $i)) &
done

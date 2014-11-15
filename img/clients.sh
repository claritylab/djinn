task=$1
numservers=$2
BASE_PORT=7999

for i in $(seq 1 $numservers);
do
  ./run-img.sh $1 $((BASE_PORT + $i)) &
done

BASE_PORT=8099
for i in $(seq 1 $numservers);
do
  ./run-img.sh $1 $((BASE_PORT + $i)) &
done

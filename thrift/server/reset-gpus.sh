numgpus=$1

for i in $(seq 1 $numgpus);
do
  sudo nvidia-smi -r --id=$((-1 + $i))
done

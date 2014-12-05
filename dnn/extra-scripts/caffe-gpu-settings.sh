# set gpu settings following Caffe optimizations
echo Script steps:
echo Turn off ECC
echo Reboot after ECC
echo Set persistence mode
echo Set max clock
for i in {0..7}
do
    echo GPU $i
    sudo nvidia-smi -i $i --ecc-config=0
    sudo nvidia-smi -i $i -r
    sudo nvidia-smi -i $i -pm 1
    sudo nvidia-smi -i $i -ac 3004,875
done

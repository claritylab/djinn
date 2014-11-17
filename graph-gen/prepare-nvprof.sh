# Use this on NVPROF logs to remove first 3 lines

dir=$1
for f in $dir/*.csv
do
    cat $f | grep -v "==" > temp.txt
    mv temp.txt $f
done

./lists.sh $dir

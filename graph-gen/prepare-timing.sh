# Use this on timing.csv to add header

header="app,plat,batch,fwd"
dir=$1
for f in $dir/*.csv
do
    echo $header > temp.txt
    cat $f >> temp.txt
    mv temp.txt $f
done

./lists.sh $dir

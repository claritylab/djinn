# Use this on timing.csv to add header

header="app,plat,batch,fwd,qpms"
dir=$1
for f in $dir/timing_*.csv
do
    echo $header > temp.txt
    cat $f >> temp.txt
    mv temp.txt $f
done

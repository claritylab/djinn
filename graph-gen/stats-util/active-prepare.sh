rm -rf *.txt
mkdir -p graphs
# for d in asr chk dig face imc ner pos
for d in imc
do
    echo $d
    ./prepare-nvprof.sh $d
    ls $d/*.csv >> active.txt
    # ./metrics-parser.py $d.txt $d
done

mv *.png graphs

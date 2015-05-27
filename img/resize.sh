INPUT=$1
SIZE=$2
DIR=$(dirname $INPUT)
IN=${INPUT##*/}
IN=${IN%.jpg}
OUTPUT=$IN-$SIZE.jpg

convert $INPUT -resize ${SIZE}x${SIZE}^ -gravity center -crop ${SIZE}x${SIZE}+0+0 $DIR/$OUTPUT

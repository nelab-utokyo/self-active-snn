set -eu

trap signalExit 2
function signalExit(){
    echo "\r"
    exit 2
}

if [ $# -le 0 ]
then
    echo "argument error: usage: bash $0 C_FILE PARALLEL(OPTION)"
    exit 1
fi
OPTION=""
if [ $# -ge 2 ]
then
    OPTION="--parallel"
fi

IS_FILE=`echo $1 | sed -e "s/config/states/" | sed -e "s/\.json/_time0000\.npz/"`

# SPONT
if [ ! -e $IS_FILE ]
then
    python snn_spont.py $1 $IS_FILE -l 0 $OPTION
else
    echo "PASS: $IS_FILE already exists"
fi

bash record.sh $1 $IS_FILE $OPTION
bash plot.sh $1 $IS_FILE $OPTION


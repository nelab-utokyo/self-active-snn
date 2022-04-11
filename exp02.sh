
trap signalExit 2
function signalExit(){
    echo "\r"
    exit 2
}

if [ $# -le 1 ]
then
    echo "argument error: usage: bash $0 C_FILE LENGTH PARALLEL(OPTION)"
    exit 1
fi
OPTION=""
if [ $# -ge 3 ]
then
    OPTION="--parallel"
fi

TIMES=`seq $2`
for TIME in ${TIMES[@]}
do
    TIME_P=`printf %04d $TIME`
    TIME_PRE=`expr $TIME - 1`
    TIME_PRE_P=`printf %04d $TIME_PRE`

    S_FILE=`echo $1 | sed -e "s/config/states/" | sed -e "s/\.json/_time${TIME_P}\.npz/"`
    IS_FILE=`echo $1 | sed -e "s/config/states/" | sed -e "s/\.json/_time${TIME_PRE_P}\.npz/"`

    if [ ! -e $S_FILE ]
    then
        python snn_spont.py $1 $S_FILE -s $IS_FILE -l 1 --seed $TIME_PRE $OPTION
    else
        echo "PASS: $S_FILE already exists"
    fi

    bash record.sh $1 $S_FILE $OPTION
    bash plot.sh $1 $S_FILE $OPTION
done

#TIME=`printf %04d $2`
#S_FILE=`echo $1 | sed -e "s/config/states/" | sed -e "s/\.json/_time${TIME}\.npz/"`
#IS_FILE=`echo $1 | sed -e "s/config/states/" | sed -e "s/\.json/_time0000\.npz/"`
#
#if [ ! -e $S_FILE ]
#then
#    python snn_spont.py $1 $S_FILE -s $IS_FILE -l $2
#else
#    echo "PASS: $S_FILE already exists"
#fi
#
#bash record_and_plot.sh $1 $S_FILE


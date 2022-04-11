
trap signalExit 2
function signalExit(){
    echo "\r"
    exit 2
}

if [ $# -ne 4 ]
then
    echo "argument error: usage: bash $0 dir length width padding"
    exit 1
fi

DIR=$1
LEN=$2
WIDTH=$3
PAD=$4

WIDTH_P=`echo $(echo $WIDTH | awk '{print $1 * 1000}')`
WIDTH_P=`printf %04d $WIDTH_P`

TIMES=()
for TIME in `seq 0 $PAD $LEN`
do
    TIME_P=`printf %04d $TIME`
    TIMES=("${TIMES[@]}" "$TIME_P")
    echo $TIME_P
done

for TIME in ${TIMES[@]}
do
    RESULT=`ls ${DIR}/0*_${TIME}.npz`
    if [ $? -ne 0 ]
    then
    continue
    fi

    echo $TIME
    OUTFILE_IFR="$DIR/ifr_W${WIDTH_P}_${TIME}.npz"
    if [ ! -e $OUTFILE_IFR ]
    then
        python ifr.py ${DIR}/0*_${TIME}.npz -o $OUTFILE_IFR
    fi
    OUTFILE_SLR="$DIR/slr_W${WIDTH_P}_${TIME}.npz"
    if [ ! -e $OUTFILE_SLR ]
    then
        python slr.py $OUTFILE_IFR $OUTFILE_SLR
    fi  
done


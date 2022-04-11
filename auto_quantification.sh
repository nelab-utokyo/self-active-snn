
trap signalExit 2
function signalExit(){
    echo "\r"
    exit 2
}

if [ $# -lt 3 ]
then
    echo "argument error: usage: bash $0 dir length width suffix"
    exit 1
fi
SUFFIX=""
if [ $# -gt 3 ]
then
    SUFFIX="_$4"
fi

DIR=$1
LEN=$2
WIDTH=$3

WIDTH_P=`echo $(echo $WIDTH | awk '{print $1 * 1000}')`
WIDTH_P=`printf %04d $WIDTH_P`

TIMES=("before$SUFFIX")
for TIME_AFTER in `seq 0 $LEN`
do
    TIME_AFTER_P=`printf after%02d$SUFFIX $TIME_AFTER`
    TIMES=("${TIMES[@]}" "${TIME_AFTER_P}")
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
        python ifr.py ${DIR}/0*_${TIME}.npz -o $OUTFILE_IFR -w $WIDTH -s $WIDTH --post 0.5
    fi  
    OUTFILE_SLR="$DIR/slr_W${WIDTH_P}_${TIME}.npz"
    if [ ! -e $OUTFILE_SLR ]
    then
        python slr.py $OUTFILE_IFR $OUTFILE_SLR
    fi  
done


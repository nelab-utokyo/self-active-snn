
trap signalExit 2
function signalExit(){
    exit 2
}

if [ $# -ne 2 ]
then
    echo "argument error: usage: bash $0 OUTDIR LENGTH"
    exit 1
fi

OUTDIR=$1
LENGTH=$2

TIME_P=`printf %04d $LENGTH`
RECORDS=(`ls ${OUTDIR}/record_600_*_time$TIME_P.npz`)
python plot_delta.py ${RECORDS[@]} -o ${OUTDIR}/deltap_time$TIME_P.pdf

